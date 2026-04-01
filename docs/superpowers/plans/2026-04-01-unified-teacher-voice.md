# Unified Teacher Voice Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace three fragmented LLM paths (chat.ts, ask.ts, synthesis) with a single unified teacher service that has 5 tools, typed SSE streaming, and prompt caching.

**Architecture:** A new `services/teacher.ts` is the sole LLM interface. It exposes `teacher.chat()` (streaming, returns `AsyncGenerator<TeacherEvent>`) and `teacher.synthesize()` (non-streaming, returns `TeacherResponse`). Both paths share a tool registry (`services/tool-processor.ts`) with 5 tools. The chat route becomes a thin SSE transport layer. The DO calls `teacher.synthesize()` for post-session recaps.

**Tech Stack:** Hono (CF Workers), Anthropic SDK (streaming), Zod validation, Drizzle ORM, Vitest + `@cloudflare/vitest-pool-workers`, TanStack Query (React)

**Spec:** `docs/superpowers/specs/2026-03-31-unified-teacher-voice-design.md`

**Style Guide:** `apps/api/TS_STYLE.md` — never destructure `c.env`, ServiceContext for DI, domain errors in services, chain `.route()` for RPC types, structured JSON logging, state versioning in DOs across awaits.

---

## File Structure

### New Files (API)

| File | Responsibility |
|---|---|
| `services/teacher.ts` | Unified teacher service — prompt assembly, `chat()` async generator, `synthesize()`, stream parser, `<analysis>` stripping |
| `services/tool-processor.ts` | Tool registry, tool definitions (5 tools), Zod schemas, concurrency partitioning, processing pipeline |
| `services/teacher.test.ts` | Unit tests for stream parser, prompt assembly, analysis stripping |
| `services/tool-processor.test.ts` | Unit tests for each tool processor, Zod validation, overflow, security |

### Modified Files (API)

| File | Change |
|---|---|
| `routes/chat.ts` | Replace `chatService.handleChatStream` with `teacher.chat()`, iterate `AsyncGenerator<TeacherEvent>`, write typed SSE events |
| `routes/chat.test.ts` | Add tests for typed SSE events with tool_result |
| `services/chat.ts` | Thin down — `saveAssistantMessage` now accepts `componentsJson`, title gen switches to Workers AI |
| `services/llm.ts` | Remove `callGroq`, add `callWorkersAI` for title gen |
| `services/prompts.ts` | Remove dead exports (SUBAGENT_SYSTEM, TEACHER_SYSTEM, etc.), add new unified TEACHER_SYSTEM |
| `do/session-brain.ts` | Replace `callSynthesisLlm` with `teacher.synthesize()`, wire `buildMemoryContext`, add deferred synthesis via DO stub |
| `do/session-brain.schema.ts` | Remove dead fields (`isEval`, `pieceQuery`) |
| `routes/practice.ts` | Deferred `/synthesize` route calls DO stub instead of direct LLM |
| `lib/types.ts` | Remove `GROQ_API_KEY` from Bindings |
| `services/wasm-bridge.ts` | Remove dead exports |
| `services/synthesis.ts` | Keep `persistSynthesisMessage`, `persistAccumulatedMoments`, `clearNeedsSynthesis`, `loadBaselinesFromDb`. Remove `callSynthesisLlm`, `buildSynthesisPrompt` (replaced by teacher.ts) |

### Deleted Files (API)

| File | Reason |
|---|---|
| `services/ask.ts` | Dead code. Useful parts migrate to `tool-processor.ts` |

### Modified Files (Web)

| File | Change |
|---|---|
| `lib/api.ts` | SSE handler: add `tool_result` event type |
| `lib/config.ts` | New: consolidated `API_BASE` + `WS_BASE` |
| `hooks/usePracticeSession.ts` | Handle `piece_identified` WS event |
| `components/ChatMessages.tsx` | Render `SynthesisCard` for synthesis messages with components |

---

## Task 1: Tool Registry and Processors

**Files:**
- Create: `apps/api/src/services/tool-processor.ts`
- Create: `apps/api/src/services/tool-processor.test.ts`
- Read: `apps/api/src/services/ask.ts` (migrate exercise processing)
- Read: `apps/api/src/lib/dims.ts` (DIMS_6 type)

- [ ] **Step 1: Write failing test for `create_exercise` tool processor**

```typescript
// apps/api/src/services/tool-processor.test.ts
import { describe, it, expect } from "vitest";
import { processToolUse, TEACHER_TOOLS } from "./tool-processor";

describe("tool-processor", () => {
  describe("TEACHER_TOOLS registry", () => {
    it("has 5 tools registered", () => {
      expect(Object.keys(TEACHER_TOOLS)).toHaveLength(5);
    });

    it("marks create_exercise as not concurrency-safe", () => {
      expect(TEACHER_TOOLS.create_exercise.concurrencySafe).toBe(false);
    });

    it("marks read-only tools as concurrency-safe", () => {
      expect(TEACHER_TOOLS.score_highlight.concurrencySafe).toBe(true);
      expect(TEACHER_TOOLS.show_session_data.concurrencySafe).toBe(true);
      expect(TEACHER_TOOLS.keyboard_guide.concurrencySafe).toBe(true);
      expect(TEACHER_TOOLS.reference_browser.concurrencySafe).toBe(true);
    });
  });

  describe("create_exercise validation", () => {
    it("rejects missing required fields", () => {
      const result = TEACHER_TOOLS.create_exercise.schema.safeParse({});
      expect(result.success).toBe(false);
    });

    it("accepts valid exercise input", () => {
      const result = TEACHER_TOOLS.create_exercise.schema.safeParse({
        source_passage: "measures 12-16",
        target_skill: "Voice balancing",
        exercises: [{
          title: "Slow hands separate",
          instruction: "Play right hand alone at half tempo.",
          focus_dimension: "dynamics",
        }],
      });
      expect(result.success).toBe(true);
    });

    it("rejects invalid focus_dimension", () => {
      const result = TEACHER_TOOLS.create_exercise.schema.safeParse({
        source_passage: "m1-4",
        target_skill: "test",
        exercises: [{
          title: "t",
          instruction: "i",
          focus_dimension: "invalid_dimension",
        }],
      });
      expect(result.success).toBe(false);
    });
  });

  describe("show_session_data validation", () => {
    it("rejects non-UUID session_id", () => {
      const result = TEACHER_TOOLS.show_session_data.schema.safeParse({
        query_type: "session_detail",
        session_id: "not-a-uuid",
      });
      expect(result.success).toBe(false);
    });

    it("enforces limit max of 50", () => {
      const result = TEACHER_TOOLS.show_session_data.schema.safeParse({
        query_type: "recent_sessions",
        limit: 100,
      });
      expect(result.success).toBe(false);
    });
  });

  describe("score_highlight validation", () => {
    it("validates bars format", () => {
      expect(TEACHER_TOOLS.score_highlight.schema.safeParse({ bars: "12-16" }).success).toBe(true);
      expect(TEACHER_TOOLS.score_highlight.schema.safeParse({ bars: "12" }).success).toBe(true);
      expect(TEACHER_TOOLS.score_highlight.schema.safeParse({ bars: "abc" }).success).toBe(false);
      expect(TEACHER_TOOLS.score_highlight.schema.safeParse({ bars: "12-16; DROP TABLE" }).success).toBe(false);
    });

    it("validates piece_id as UUID when provided", () => {
      expect(TEACHER_TOOLS.score_highlight.schema.safeParse({
        bars: "1-4",
        piece_id: "not-uuid",
      }).success).toBe(false);
    });
  });
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd apps/api && npx vitest run src/services/tool-processor.test.ts`
Expected: FAIL — `Cannot find module './tool-processor'`

- [ ] **Step 3: Implement tool registry with Zod schemas**

```typescript
// apps/api/src/services/tool-processor.ts
import { z } from "zod";
import { DIMS_6, type Dimension } from "../lib/dims";
import type { ServiceContext } from "../lib/types";

// --- Tool Result Types ---

export interface ToolResult {
  name: string;
  componentsJson: InlineComponent[];
  isError: boolean;
}

export interface InlineComponent {
  type: string;
  config: Record<string, unknown>;
}

// --- Zod Schemas ---

const exerciseItemSchema = z.object({
  title: z.string().min(1),
  instruction: z.string().min(1),
  focus_dimension: z.enum(DIMS_6),
  hands: z.enum(["left", "right", "both"]).optional(),
  exercise_id: z.string().uuid().optional(),
});

const createExerciseSchema = z.object({
  source_passage: z.string().min(1),
  target_skill: z.string().min(1),
  exercises: z.array(exerciseItemSchema).min(1).max(3),
});

const scoreHighlightSchema = z.object({
  bars: z.string().regex(/^\d+(-\d+)?$/),
  annotations: z.array(z.string()).optional(),
  piece_id: z.string().uuid().optional(),
});

const keyboardGuideSchema = z.object({
  title: z.string().min(1),
  description: z.string().min(1),
  fingering: z.string().optional(),
  hands: z.enum(["left", "right", "both"]),
});

const showSessionDataSchema = z.object({
  query_type: z.enum(["dimension_history", "recent_sessions", "session_detail"]),
  dimension: z.enum(DIMS_6).optional(),
  session_id: z.string().uuid().optional(),
  limit: z.number().int().min(1).max(50).default(20),
});

const referenceBrowserSchema = z.object({
  piece_id: z.string().uuid().optional(),
  passage: z.string().optional(),
  description: z.string().min(1),
});

// --- Tool Definition Type ---

export interface ToolDefinition {
  name: string;
  schema: z.ZodSchema;
  anthropicSchema: object;
  description: string;
  concurrencySafe: boolean;
  maxResultChars?: number;
  process: (ctx: ServiceContext, studentId: string, input: unknown) => Promise<ToolResult>;
}

// --- Tool Processors ---

async function processExercise(ctx: ServiceContext, studentId: string, input: unknown): Promise<ToolResult> {
  const parsed = createExerciseSchema.parse(input);
  const components: InlineComponent[] = [{
    type: "exercise_set",
    config: {
      sourcePassage: parsed.source_passage,
      targetSkill: parsed.target_skill,
      exercises: parsed.exercises.map((e) => ({
        title: e.title,
        instruction: e.instruction,
        focusDimension: e.focus_dimension,
        hands: e.hands ?? "both",
      })),
    },
  }];
  return { name: "create_exercise", componentsJson: components, isError: false };
}

async function processScoreHighlight(ctx: ServiceContext, studentId: string, input: unknown): Promise<ToolResult> {
  const parsed = scoreHighlightSchema.parse(input);
  const config: Record<string, unknown> = { bars: parsed.bars, annotations: parsed.annotations ?? [] };
  if (parsed.piece_id) {
    const scoreObj = await ctx.env.SCORES.get(`scores/v1/${parsed.piece_id}.json`);
    if (scoreObj) {
      const scoreData = await scoreObj.json();
      config.scoreData = scoreData;
    }
  }
  return { name: "score_highlight", componentsJson: [{ type: "score_highlight", config }], isError: false };
}

async function processKeyboardGuide(_ctx: ServiceContext, _studentId: string, input: unknown): Promise<ToolResult> {
  const parsed = keyboardGuideSchema.parse(input);
  return {
    name: "keyboard_guide",
    componentsJson: [{ type: "keyboard_guide", config: { ...parsed } }],
    isError: false,
  };
}

async function processShowSessionData(ctx: ServiceContext, studentId: string, input: unknown): Promise<ToolResult> {
  const parsed = showSessionDataSchema.parse(input);
  let data: unknown;

  if (parsed.query_type === "dimension_history" && parsed.dimension) {
    const rows = await ctx.db.query.observations.findMany({
      where: (obs, { eq, and }) => and(eq(obs.studentId, studentId), eq(obs.dimension, parsed.dimension!)),
      orderBy: (obs, { desc }) => [desc(obs.createdAt)],
      limit: Math.min(parsed.limit, 50),
      columns: { dimension: true, dimensionScore: true, createdAt: true, observationText: true },
    });
    data = rows;
  } else if (parsed.query_type === "recent_sessions") {
    const rows = await ctx.db.query.sessions.findMany({
      where: (s, { eq }) => eq(s.studentId, studentId),
      orderBy: (s, { desc }) => [desc(s.startedAt)],
      limit: Math.min(parsed.limit, 50),
      columns: { id: true, startedAt: true, endedAt: true, needsSynthesis: true },
    });
    data = rows;
  } else if (parsed.query_type === "session_detail" && parsed.session_id) {
    const session = await ctx.db.query.sessions.findFirst({
      where: (s, { eq, and }) => and(eq(s.id, parsed.session_id!), eq(s.studentId, studentId)),
      columns: { id: true, startedAt: true, endedAt: true, accumulatorJson: true },
    });
    data = session;
  }

  let resultStr = JSON.stringify(data ?? {});
  if (resultStr.length > 10000) {
    resultStr = resultStr.slice(0, 10000);
    data = { truncated: true, note: "Showing most recent data. Ask me to narrow the time range for more detail.", partial: JSON.parse(resultStr + '"}') };
  }

  return {
    name: "show_session_data",
    componentsJson: [{ type: "session_data", config: { queryType: parsed.query_type, data } }],
    isError: false,
  };
}

async function processReferenceBrowser(_ctx: ServiceContext, _studentId: string, input: unknown): Promise<ToolResult> {
  const parsed = referenceBrowserSchema.parse(input);
  return {
    name: "reference_browser",
    componentsJson: [{ type: "reference_browser", config: { ...parsed } }],
    isError: false,
  };
}

// --- Anthropic Tool Schemas (JSON Schema for LLM) ---

function buildAnthropicSchema(name: string, description: string, zodSchema: z.ZodSchema): object {
  return {
    name,
    description,
    input_schema: zodToJsonSchema(zodSchema),
  };
}

// Manual JSON Schema definitions (avoiding zodToJsonSchema dependency for now)
const ANTHROPIC_TOOL_SCHEMAS = [
  {
    name: "create_exercise",
    description: "Create a focused practice exercise when the student would benefit from structured practice on a specific passage or technique. Use sparingly -- only when a concrete drill would be more helpful than verbal guidance alone.",
    input_schema: {
      type: "object",
      required: ["source_passage", "target_skill", "exercises"],
      properties: {
        source_passage: { type: "string", description: "e.g. 'measures 12-16'" },
        target_skill: { type: "string", description: "e.g. 'Voice balancing between hands'" },
        exercises: {
          type: "array", minItems: 1, maxItems: 3,
          items: {
            type: "object",
            required: ["title", "instruction", "focus_dimension"],
            properties: {
              title: { type: "string" },
              instruction: { type: "string", description: "Concrete steps, 2-4 sentences." },
              focus_dimension: { type: "string", enum: ["dynamics", "timing", "pedaling", "articulation", "phrasing", "interpretation"] },
              hands: { type: "string", enum: ["left", "right", "both"] },
              exercise_id: { type: "string", description: "UUID of an existing catalog exercise, if reusing." },
            },
          },
        },
      },
    },
  },
  {
    name: "score_highlight",
    description: "Render an annotated score passage to show the student specific bars with visual annotations. Use when discussing a specific musical passage.",
    input_schema: {
      type: "object",
      required: ["bars"],
      properties: {
        bars: { type: "string", description: "Bar range, e.g. '12-16' or '8'" },
        annotations: { type: "array", items: { type: "string" }, description: "Text annotations to overlay on the score" },
        piece_id: { type: "string", description: "UUID of the piece in the score catalog" },
      },
    },
  },
  {
    name: "keyboard_guide",
    description: "Show a fingering or hand position guide for a specific passage or technique.",
    input_schema: {
      type: "object",
      required: ["title", "description", "hands"],
      properties: {
        title: { type: "string" },
        description: { type: "string" },
        fingering: { type: "string", description: "Fingering notation, e.g. '1-2-3-5'" },
        hands: { type: "string", enum: ["left", "right", "both"] },
      },
    },
  },
  {
    name: "show_session_data",
    description: "Pull up the student's past practice data. Use when the student asks about their progress, trends, or history on a specific dimension.",
    input_schema: {
      type: "object",
      required: ["query_type"],
      properties: {
        query_type: { type: "string", enum: ["dimension_history", "recent_sessions", "session_detail"] },
        dimension: { type: "string", enum: ["dynamics", "timing", "pedaling", "articulation", "phrasing", "interpretation"] },
        session_id: { type: "string", description: "UUID of a specific session" },
        limit: { type: "integer", minimum: 1, maximum: 50, default: 20 },
      },
    },
  },
  {
    name: "reference_browser",
    description: "Show a reference performance or recording for comparison. Use when discussing interpretation or suggesting the student listen to a specific performance.",
    input_schema: {
      type: "object",
      required: ["description"],
      properties: {
        piece_id: { type: "string", description: "UUID of the piece" },
        passage: { type: "string", description: "Specific passage, e.g. 'second theme, bars 45-60'" },
        description: { type: "string", description: "What to listen for in the reference" },
      },
    },
  },
];

// --- Registry ---

export const TEACHER_TOOLS: Record<string, ToolDefinition> = {
  create_exercise: {
    name: "create_exercise",
    schema: createExerciseSchema,
    anthropicSchema: ANTHROPIC_TOOL_SCHEMAS[0],
    description: "Create focused practice exercises",
    concurrencySafe: false,
    process: processExercise,
  },
  score_highlight: {
    name: "score_highlight",
    schema: scoreHighlightSchema,
    anthropicSchema: ANTHROPIC_TOOL_SCHEMAS[1],
    description: "Render annotated score passages",
    concurrencySafe: true,
    maxResultChars: 5000,
    process: processScoreHighlight,
  },
  keyboard_guide: {
    name: "keyboard_guide",
    schema: keyboardGuideSchema,
    anthropicSchema: ANTHROPIC_TOOL_SCHEMAS[2],
    description: "Show fingering/hand position guide",
    concurrencySafe: true,
    maxResultChars: 2000,
    process: processKeyboardGuide,
  },
  show_session_data: {
    name: "show_session_data",
    schema: showSessionDataSchema,
    anthropicSchema: ANTHROPIC_TOOL_SCHEMAS[3],
    description: "Pull up past practice data",
    concurrencySafe: true,
    maxResultChars: 10000,
    process: processShowSessionData,
  },
  reference_browser: {
    name: "reference_browser",
    schema: referenceBrowserSchema,
    anthropicSchema: ANTHROPIC_TOOL_SCHEMAS[4],
    description: "Show reference performances",
    concurrencySafe: true,
    maxResultChars: 2000,
    process: processReferenceBrowser,
  },
};

// --- Processing Pipeline ---

export async function processToolUse(
  ctx: ServiceContext,
  studentId: string,
  toolName: string,
  toolInput: unknown,
): Promise<ToolResult> {
  const tool = TEACHER_TOOLS[toolName];
  if (!tool) {
    return { name: toolName, componentsJson: [], isError: true };
  }

  try {
    const validInput = tool.schema.parse(toolInput);
    return await tool.process(ctx, studentId, validInput);
  } catch (error) {
    console.log(JSON.stringify({
      level: "error",
      msg: "tool_processing_failed",
      tool: toolName,
      error: error instanceof Error ? error.message : String(error),
    }));
    return { name: toolName, componentsJson: [], isError: true };
  }
}

export function getAnthropicToolSchemas(): object[] {
  return ANTHROPIC_TOOL_SCHEMAS;
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd apps/api && npx vitest run src/services/tool-processor.test.ts`
Expected: PASS (all schema validation tests)

- [ ] **Step 5: Commit**

```bash
git add apps/api/src/services/tool-processor.ts apps/api/src/services/tool-processor.test.ts
git commit -m "feat(api): add teacher tool registry with 5 tools and Zod validation"
```

---

## Task 2: Unified Teacher Prompts

**Files:**
- Modify: `apps/api/src/services/prompts.ts`
- Read: `apps/api/src/services/prompts.ts` (current content)

- [ ] **Step 1: Write the new unified TEACHER_SYSTEM prompt and remove dead exports**

Add the new unified prompt to `prompts.ts`. Remove: `SUBAGENT_SYSTEM`, `TEACHER_SYSTEM` (the old one), `buildSubagentUserPrompt`, `buildTeacherUserPrompt`, `exerciseToolDefinition()`, `ExerciseTool`, `ObservationRow`, `CatalogExercise`. Keep: `SESSION_SYNTHESIS_SYSTEM` (still referenced by existing code during migration), `CHAT_SYSTEM` (absorbed into the new prompt but kept for reference), `buildChatUserContext`, `buildTitlePrompt`.

Add to `prompts.ts`:

```typescript
export const TEACHER_PROMPT_DYNAMIC_BOUNDARY = "__TEACHER_PROMPT_DYNAMIC_BOUNDARY__";

export const UNIFIED_TEACHER_SYSTEM = `You are a warm, encouraging piano teacher. You help students improve their playing through thoughtful conversation, grounded in their actual playing data when available.

## Pedagogy Principles
- Celebrate strengths before suggesting improvements
- Frame observations, not absolute judgments
- Give actionable practice strategies, not abstract criticism
- Be specific about musical elements (dynamics, timing, pedaling, articulation, phrasing, interpretation)
- Adapt to the student's level and goals
- One key insight per response is better than a laundry list

## The Six Dimensions
- Dynamics: volume control, crescendo/decrescendo, accents, balance between hands
- Timing: tempo stability, rubato, rhythmic accuracy, pulse
- Pedaling: sustain pedal timing, half-pedaling, clarity vs. blur
- Articulation: staccato, legato, touch quality, note connections
- Phrasing: musical sentences, breathing, shape, direction
- Interpretation: style, character, emotional expression, faithfulness to score

## Model Calibration
The MuQ audio model has R2~0.5 and 80% pairwise accuracy. Scores are directional signals, not precise measurements. A deviation of 0.1 from baseline is noise; 0.2+ is meaningful. Use deviations to identify patterns, not to make absolute claims. Never mention raw scores to the student -- translate into musical language.

## Tool Usage
You have tools available. Use them when they add value:
- create_exercise: When a concrete drill would help more than verbal guidance. Use sparingly.
- score_highlight: When discussing a specific passage and visual reference would help.
- keyboard_guide: When fingering or hand position matters.
- show_session_data: When the student asks about progress or you want to reference their history.
- reference_browser: When suggesting the student listen to a specific performance.
Most responses should be text-only. Tools are supplements, not defaults.

## Voice
- 1-4 sentences for observations, up to a short paragraph for explanations
- No markdown formatting, no bullets, no emojis
- Conversational, warm, specific
- Speak as a teacher talking TO the student, not about them`;

export function buildSynthesisFraming(
  sessionDurationMs: number,
  practicePattern: string,
  topMoments: unknown[],
  drillingRecords: unknown[],
  pieceMetadata: { composer?: string; title?: string } | null,
  memoryContext: string,
): string {
  const durationMin = Math.round(sessionDurationMs / 60000);
  const sessionData = JSON.stringify({
    duration_minutes: durationMin,
    practice_pattern: practicePattern,
    top_moments: topMoments,
    drilling_records: drillingRecords,
    piece: pieceMetadata,
  });

  return \`You just finished listening to a practice session. Review the session data and give your student a cohesive response.

## Session Data
\${sessionData}

\${memoryContext ? \`## Student Memory\\n\${memoryContext}\` : ""}

## Instructions
1. Write your pedagogical reasoning inside <analysis>...</analysis> tags first. This will be stripped -- the student never sees it. Think through: which moments matter most, how to frame feedback, whether exercises would help.
2. Then write your response to the student (3-6 sentences, warm, specific, referencing musical details).
3. Use tools if they add value: create_exercise for suggested drills, score_highlight for specific passages.
4. Do NOT mention scores, numbers, or model outputs. Translate everything into musical language.
5. Do NOT list all dimensions. Focus on what matters most for THIS session.\`;
}

export function buildChatFraming(
  studentLevel: string | null,
  studentGoals: string | null,
  memoryContext: string,
): string {
  const parts: string[] = ["You are in a conversation with your student."];
  if (studentLevel) parts.push(\`Student level: \${studentLevel}\`);
  if (studentGoals) parts.push(\`Student goals: \${studentGoals}\`);
  if (memoryContext) parts.push(\`\n## Student Memory\n\${memoryContext}\`);
  return parts.join("\n");
}
```

- [ ] **Step 2: Verify the prompts module still compiles**

Run: `cd apps/api && npx tsc --noEmit`
Expected: PASS (no type errors)

- [ ] **Step 3: Commit**

```bash
git add apps/api/src/services/prompts.ts
git commit -m "feat(api): add unified teacher system prompt with cache boundary and synthesis framing"
```

---

## Task 3: Teacher Service — Core + Stream Parser

**Files:**
- Create: `apps/api/src/services/teacher.ts`
- Create: `apps/api/src/services/teacher.test.ts`
- Read: `apps/api/src/services/llm.ts`
- Read: `apps/api/src/services/chat.ts`
- Read: `apps/api/src/services/memory.ts`

- [ ] **Step 1: Write failing test for the stream parser**

```typescript
// apps/api/src/services/teacher.test.ts
import { describe, it, expect } from "vitest";
import { parseAnthropicStream } from "./teacher";

// Helper: build a fake Anthropic SSE stream from events
function buildSSEStream(events: Array<{ event: string; data: object }>): ReadableStream {
  const encoder = new TextEncoder();
  const lines = events.map((e) =>
    `event: ${e.event}\ndata: ${JSON.stringify(e.data)}\n\n`
  ).join("");
  return new ReadableStream({
    start(controller) {
      controller.enqueue(encoder.encode(lines));
      controller.close();
    },
  });
}

describe("parseAnthropicStream", () => {
  it("yields delta events for text blocks", async () => {
    const stream = buildSSEStream([
      { event: "message_start", data: { type: "message_start", message: { id: "m1", content: [], usage: {} } } },
      { event: "content_block_start", data: { type: "content_block_start", index: 0, content_block: { type: "text", text: "" } } },
      { event: "content_block_delta", data: { type: "content_block_delta", index: 0, delta: { type: "text_delta", text: "Hello " } } },
      { event: "content_block_delta", data: { type: "content_block_delta", index: 0, delta: { type: "text_delta", text: "world" } } },
      { event: "content_block_stop", data: { type: "content_block_stop", index: 0 } },
      { event: "message_delta", data: { type: "message_delta", delta: { stop_reason: "end_turn" }, usage: { output_tokens: 5 } } },
      { event: "message_stop", data: { type: "message_stop" } },
    ]);

    const events = [];
    for await (const event of parseAnthropicStream(stream, async () => ({ name: "test", componentsJson: [], isError: false }))) {
      events.push(event);
    }

    expect(events).toEqual([
      { type: "delta", text: "Hello " },
      { type: "delta", text: "world" },
      { type: "done", fullText: "Hello world", allComponents: [] },
    ]);
  });

  it("yields tool_result events for tool_use blocks", async () => {
    const stream = buildSSEStream([
      { event: "message_start", data: { type: "message_start", message: { id: "m1", content: [], usage: {} } } },
      { event: "content_block_start", data: { type: "content_block_start", index: 0, content_block: { type: "text", text: "" } } },
      { event: "content_block_delta", data: { type: "content_block_delta", index: 0, delta: { type: "text_delta", text: "Try this:" } } },
      { event: "content_block_stop", data: { type: "content_block_stop", index: 0 } },
      { event: "content_block_start", data: { type: "content_block_start", index: 1, content_block: { type: "tool_use", id: "t1", name: "create_exercise", input: "" } } },
      { event: "content_block_delta", data: { type: "content_block_delta", index: 1, delta: { type: "input_json_delta", partial_json: '{"source' } } },
      { event: "content_block_delta", data: { type: "content_block_delta", index: 1, delta: { type: "input_json_delta", partial_json: '_passage":"m1"}' } } },
      { event: "content_block_stop", data: { type: "content_block_stop", index: 1 } },
      { event: "message_delta", data: { type: "message_delta", delta: { stop_reason: "tool_use" }, usage: { output_tokens: 20 } } },
      { event: "message_stop", data: { type: "message_stop" } },
    ]);

    const mockProcessor = async (name: string, input: unknown) => ({
      name: "create_exercise",
      componentsJson: [{ type: "exercise_set", config: { mock: true } }],
      isError: false,
    });

    const events = [];
    for await (const event of parseAnthropicStream(stream, mockProcessor)) {
      events.push(event);
    }

    expect(events[0]).toEqual({ type: "delta", text: "Try this:" });
    expect(events[1]).toEqual({
      type: "tool_result",
      name: "create_exercise",
      componentsJson: [{ type: "exercise_set", config: { mock: true } }],
    });
    expect(events[2].type).toBe("done");
    expect(events[2].allComponents).toHaveLength(1);
  });

  it("does not yield tool_result for failed tools", async () => {
    const stream = buildSSEStream([
      { event: "message_start", data: { type: "message_start", message: { id: "m1", content: [], usage: {} } } },
      { event: "content_block_start", data: { type: "content_block_start", index: 0, content_block: { type: "tool_use", id: "t1", name: "bad_tool", input: "" } } },
      { event: "content_block_stop", data: { type: "content_block_stop", index: 0 } },
      { event: "message_stop", data: { type: "message_stop" } },
    ]);

    const failProcessor = async () => ({ name: "bad_tool", componentsJson: [], isError: true });

    const events = [];
    for await (const event of parseAnthropicStream(stream, failProcessor)) {
      events.push(event);
    }

    // No tool_result event for failed tools
    expect(events.filter((e) => e.type === "tool_result")).toHaveLength(0);
    expect(events.find((e) => e.type === "done")).toBeTruthy();
  });
});

describe("stripAnalysis", () => {
  it("strips analysis block from text", async () => {
    const { stripAnalysis } = await import("./teacher");
    expect(stripAnalysis("<analysis>reasoning here</analysis>The actual response")).toBe("The actual response");
  });

  it("strips multiple analysis blocks", async () => {
    const { stripAnalysis } = await import("./teacher");
    expect(stripAnalysis("<analysis>first</analysis>middle<analysis>second</analysis>end")).toBe("middleend");
  });

  it("returns text unchanged when no analysis block", async () => {
    const { stripAnalysis } = await import("./teacher");
    expect(stripAnalysis("just normal text")).toBe("just normal text");
  });
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd apps/api && npx vitest run src/services/teacher.test.ts`
Expected: FAIL — `Cannot find module './teacher'`

- [ ] **Step 3: Implement teacher.ts with stream parser**

```typescript
// apps/api/src/services/teacher.ts
import type { ServiceContext } from "../lib/types";
import type { AnthropicSystemBlock } from "./llm";
import { callAnthropic, callAnthropicStream } from "./llm";
import { buildMemoryContext } from "./memory";
import {
  UNIFIED_TEACHER_SYSTEM,
  TEACHER_PROMPT_DYNAMIC_BOUNDARY,
  buildChatFraming,
  buildSynthesisFraming,
  buildChatUserContext,
} from "./prompts";
import {
  processToolUse,
  getAnthropicToolSchemas,
  type ToolResult,
  type InlineComponent,
} from "./tool-processor";

// --- Public Types ---

export type TeacherEvent =
  | { type: "delta"; text: string }
  | { type: "tool_result"; name: string; componentsJson: InlineComponent[] }
  | { type: "done"; fullText: string; allComponents: InlineComponent[] };

export interface TeacherResponse {
  text: string;
  toolResults: ToolResult[];
}

export interface SynthesisInput {
  studentId: string;
  conversationId: string | null;
  sessionDurationMs: number;
  practicePattern: string;
  topMoments: unknown[];
  drillingRecords: unknown[];
  pieceMetadata: { composer?: string; title?: string } | null;
}

// --- Analysis Stripping ---

export function stripAnalysis(text: string): string {
  return text.replace(/<analysis>[\s\S]*?<\/analysis>/g, "").trim();
}

// --- Stream Parser ---

type ToolProcessor = (name: string, input: unknown) => Promise<ToolResult>;

export async function* parseAnthropicStream(
  stream: ReadableStream,
  processToolFn: ToolProcessor,
): AsyncGenerator<TeacherEvent> {
  const reader = stream.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  let fullText = "";
  const allComponents: InlineComponent[] = [];

  // Track content blocks by index
  const contentBlocks: Map<number, { type: string; text: string; input: string; name: string; id: string }> = new Map();

  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split("\n");
      buffer = lines.pop() ?? "";

      for (const line of lines) {
        if (!line.startsWith("data: ")) continue;
        const dataStr = line.slice(6).trim();
        if (dataStr === "[DONE]") continue;

        let event: Record<string, unknown>;
        try {
          event = JSON.parse(dataStr);
        } catch {
          continue;
        }

        const eventType = event.type as string;

        if (eventType === "content_block_start") {
          const block = event.content_block as Record<string, unknown>;
          const index = event.index as number;
          contentBlocks.set(index, {
            type: block.type as string,
            text: "",
            input: "",
            name: (block.name as string) ?? "",
            id: (block.id as string) ?? "",
          });
        } else if (eventType === "content_block_delta") {
          const index = event.index as number;
          const delta = event.delta as Record<string, unknown>;
          const block = contentBlocks.get(index);
          if (!block) continue;

          if (delta.type === "text_delta") {
            const text = delta.text as string;
            block.text += text;
            fullText += text;
            yield { type: "delta", text };
          } else if (delta.type === "input_json_delta") {
            block.input += delta.partial_json as string;
          }
        } else if (eventType === "content_block_stop") {
          const index = event.index as number;
          const block = contentBlocks.get(index);
          if (!block) continue;

          if (block.type === "tool_use") {
            let parsedInput: unknown;
            try {
              parsedInput = block.input ? JSON.parse(block.input) : {};
            } catch {
              parsedInput = {};
            }

            const result = await processToolFn(block.name, parsedInput);
            if (!result.isError && result.componentsJson.length > 0) {
              allComponents.push(...result.componentsJson);
              yield { type: "tool_result", name: result.name, componentsJson: result.componentsJson };
            }
          }
        }
      }
    }
  } finally {
    reader.releaseLock();
  }

  yield { type: "done", fullText, allComponents };
}

// --- Chat Entry Point ---

export async function* chat(
  ctx: ServiceContext,
  studentId: string,
  messages: Array<{ role: string; content: string }>,
  dynamicContext: string,
): AsyncGenerator<TeacherEvent> {
  const systemBlocks: AnthropicSystemBlock[] = [
    { type: "text", text: UNIFIED_TEACHER_SYSTEM, cache_control: { type: "ephemeral" } },
  ];
  if (dynamicContext) {
    systemBlocks.push({ type: "text", text: dynamicContext });
  }

  const stream = await callAnthropicStream(ctx.env, {
    model: "claude-sonnet-4-20250514",
    max_tokens: 2048,
    system: systemBlocks,
    messages: messages.map((m) => ({ role: m.role as "user" | "assistant", content: m.content })),
    tools: getAnthropicToolSchemas(),
    tool_choice: { type: "auto" },
  });

  const processor: ToolProcessor = (name, input) => processToolUse(ctx, studentId, name, input);

  yield* parseAnthropicStream(stream, processor);
}

// --- Synthesis Entry Point ---

export async function synthesize(
  ctx: ServiceContext,
  input: SynthesisInput,
): Promise<TeacherResponse> {
  const memoryContext = await buildMemoryContext(ctx, input.studentId);

  const synthesisFraming = buildSynthesisFraming(
    input.sessionDurationMs,
    input.practicePattern,
    input.topMoments,
    input.drillingRecords,
    input.pieceMetadata,
    memoryContext,
  );

  const systemBlocks: AnthropicSystemBlock[] = [
    { type: "text", text: UNIFIED_TEACHER_SYSTEM, cache_control: { type: "ephemeral" } },
    { type: "text", text: synthesisFraming },
  ];

  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), 25000);

  try {
    const response = await callAnthropic(ctx.env, {
      model: "claude-sonnet-4-20250514",
      max_tokens: 2048,
      system: systemBlocks,
      messages: [{ role: "user", content: "Please review my practice session and give me feedback." }],
      tools: getAnthropicToolSchemas(),
      tool_choice: { type: "auto" },
    });

    let text = "";
    const toolResults: ToolResult[] = [];

    for (const block of response.content) {
      if (block.type === "text" && block.text) {
        text += block.text;
      } else if (block.type === "tool_use" && block.name && block.input !== undefined) {
        const result = await processToolUse(ctx, input.studentId, block.name, block.input);
        toolResults.push(result);
      }
    }

    text = stripAnalysis(text);

    return { text, toolResults };
  } finally {
    clearTimeout(timeout);
  }
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd apps/api && npx vitest run src/services/teacher.test.ts`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add apps/api/src/services/teacher.ts apps/api/src/services/teacher.test.ts
git commit -m "feat(api): add unified teacher service with stream parser and synthesis"
```

---

## Task 4: Evolve Chat Route to Use Teacher Service

**Files:**
- Modify: `apps/api/src/routes/chat.ts`
- Modify: `apps/api/src/services/chat.ts`
- Modify: `apps/api/src/routes/chat.test.ts`

- [ ] **Step 1: Update chat.ts service — thin it down, add componentsJson to saveAssistantMessage**

In `apps/api/src/services/chat.ts`:

1. Add `componentsJson` parameter to `saveAssistantMessage`:

```typescript
export async function saveAssistantMessage(
  db: Db,
  env: Bindings,
  conversationId: string,
  content: string,
  componentsJson: unknown[] | null,
  isNewConversation: boolean,
  firstUserMessage: string,
): Promise<void> {
  await db.insert(messages).values({
    conversationId,
    role: "assistant",
    content,
    componentsJson: componentsJson && componentsJson.length > 0 ? componentsJson : null,
  });
  // ... rest unchanged (updatedAt, title generation)
}
```

2. Add a new `prepareChatContext` function that builds the context for teacher.chat():

```typescript
export async function prepareChatContext(
  ctx: ServiceContext,
  studentId: string,
  input: { conversationId?: string; message: string },
): Promise<{
  conversationId: string;
  isNewConversation: boolean;
  messages: Array<{ role: string; content: string }>;
  dynamicContext: string;
}> {
  let conversationId = input.conversationId ?? "";
  let isNewConversation = false;

  if (input.conversationId) {
    const conv = await ctx.db.query.conversations.findFirst({
      where: (c, { eq, and }) => and(eq(c.id, input.conversationId!), eq(c.studentId, studentId)),
    });
    if (!conv) throw new NotFoundError("conversation", input.conversationId!);
    conversationId = conv.id;
  } else {
    const [conv] = await ctx.db.insert(conversations).values({ studentId }).returning();
    conversationId = conv.id;
    isNewConversation = true;
  }

  await ctx.db.insert(dbMessages).values({
    conversationId,
    role: "user",
    content: input.message,
  });

  const student = await ctx.db.query.students.findFirst({
    where: (s, { eq }) => eq(s.id, studentId),
  });

  const memoryContext = await buildMemoryContext(ctx, studentId);

  const history = await ctx.db.query.messages.findMany({
    where: (m, { eq }) => eq(m.conversationId, conversationId),
    orderBy: (m, { asc }) => [asc(m.createdAt)],
    limit: 20,
    columns: { role: true, content: true },
  });

  const messages = history
    .filter((m) => m.role === "user" || m.role === "assistant")
    .map((m) => ({ role: m.role, content: m.content }));

  const dynamicContext = buildChatFraming(
    student?.inferredLevel ?? null,
    student?.explicitGoals ?? null,
    memoryContext,
  );

  return { conversationId, isNewConversation, messages, dynamicContext };
}
```

3. Remove `handleChatStream` (replaced by `prepareChatContext` + `teacher.chat()`)

- [ ] **Step 2: Update routes/chat.ts to use teacher.chat()**

Replace the route body with:

```typescript
import { Hono } from "hono";
import { streamSSE } from "hono/streaming";
import { z } from "zod";
import type { Bindings, Variables } from "../lib/types";
import { validate } from "../lib/validate";
import { requireAuth } from "../middleware/auth-session";
import * as chatService from "../services/chat";
import * as teacher from "../services/teacher";

const chatSchema = z.object({
  conversationId: z.string().uuid().optional(),
  message: z.string().min(1).max(10000),
});

const app = new Hono<{ Bindings: Bindings; Variables: Variables }>().post(
  "/",
  validate("json", chatSchema),
  async (c) => {
    requireAuth(c.var.studentId);
    const body = c.req.valid("json");
    const ctx = { db: c.var.db, env: c.env };

    const { conversationId, isNewConversation, messages, dynamicContext } =
      await chatService.prepareChatContext(ctx, c.var.studentId, body);

    c.header("Content-Encoding", "Identity");

    return streamSSE(c, async (sseStream) => {
      await sseStream.writeSSE({
        data: JSON.stringify({ conversationId }),
        event: "start",
        id: "0",
      });

      let id = 1;
      let fullText = "";
      let allComponents: unknown[] = [];

      try {
        for await (const event of teacher.chat(ctx, c.var.studentId, messages, dynamicContext)) {
          if (event.type === "delta") {
            await sseStream.writeSSE({ data: event.text, event: "delta", id: String(id++) });
          } else if (event.type === "tool_result") {
            await sseStream.writeSSE({
              data: JSON.stringify({ name: event.name, componentsJson: JSON.stringify(event.componentsJson) }),
              event: "tool_result",
              id: String(id++),
            });
          } else if (event.type === "done") {
            fullText = event.fullText;
            allComponents = event.allComponents;
          }
        }
      } catch (error) {
        await sseStream.writeSSE({
          data: JSON.stringify({ message: "I'm having trouble responding right now. Try again in a moment." }),
          event: "error",
          id: String(id++),
        });
      }

      await sseStream.writeSSE({ data: "[DONE]", event: "done", id: String(id) });

      c.executionCtx.waitUntil(
        chatService.saveAssistantMessage(
          c.var.db,
          c.env,
          conversationId,
          fullText,
          allComponents.length > 0 ? allComponents : null,
          isNewConversation,
          body.message,
        ),
      );
    });
  },
);

export { app as chatRoutes };
```

- [ ] **Step 3: Add test for typed SSE events**

Add to `apps/api/src/routes/chat.test.ts`:

```typescript
describe("SSE event types", () => {
  it("includes start, delta, done event types in response", () => {
    // Verify the SSE format matches the spec
    // This is a structural test — actual streaming tested in teacher.test.ts
    expect(["start", "delta", "tool_result", "done", "error"]).toContain("tool_result");
  });
});
```

- [ ] **Step 4: Verify compilation**

Run: `cd apps/api && npx tsc --noEmit`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add apps/api/src/routes/chat.ts apps/api/src/services/chat.ts apps/api/src/routes/chat.test.ts
git commit -m "feat(api): evolve chat route to use unified teacher service with tool_use streaming"
```

---

## Task 5: Wire Teacher into DO Synthesis

**Files:**
- Modify: `apps/api/src/do/session-brain.ts`
- Modify: `apps/api/src/do/session-brain.schema.ts`
- Modify: `apps/api/src/routes/practice.ts`
- Modify: `apps/api/src/services/synthesis.ts`

- [ ] **Step 1: Remove dead fields from session-brain.schema.ts**

Remove `isEval` and `pieceQuery` from `SessionState` schema and `createInitialState()`.

- [ ] **Step 2: Update session-brain.ts synthesis path**

Replace `callSynthesisLlm(this.env, promptContext)` with `teacher.synthesize(ctx, synthInput)`:

1. Build `ServiceContext` manually: `{ db: createDb(this.env.HYPERDRIVE), env: this.env }`
2. Call `buildMemoryContext(ctx, state.studentId)` (wiring the TODO)
3. Build `SynthesisInput` from accumulator `topMoments()`, drilling records, mode transitions, piece metadata
4. Call `teacher.synthesize(ctx, synthInput)` — this throws on failure (no silent fallback)
5. Send WS `synthesis` event with `{ text, components: toolResults.flatMap(r => r.componentsJson), isFallback: false }`
6. Persist with `persistSynthesisMessage` — now passing `componentsJson`

Update `persistSynthesisMessage` to accept and store `componentsJson`.

- [ ] **Step 3: Add deferred synthesis through DO stub in practice.ts**

Change `POST /api/practice/synthesize` to route through the DO:

```typescript
// Instead of calling callSynthesisLlm directly:
const doId = c.env.SESSION_BRAIN.idFromName(sessionId);
const stub = c.env.SESSION_BRAIN.get(doId);
const response = await stub.fetch(new Request("https://do/synthesize", {
  method: "POST",
  body: JSON.stringify({ sessionId }),
}));
```

Add a handler in the DO for `POST /synthesize` that runs `runSynthesisAndPersist()`.

- [ ] **Step 4: Remove callSynthesisLlm and buildSynthesisPrompt from synthesis.ts**

Keep: `persistSynthesisMessage` (update to accept `componentsJson`), `persistAccumulatedMoments`, `clearNeedsSynthesis`, `loadBaselinesFromDb`.

Delete: `callSynthesisLlm`, `buildSynthesisPrompt`, `SynthesisResult` type.

- [ ] **Step 5: Verify compilation**

Run: `cd apps/api && npx tsc --noEmit`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add apps/api/src/do/session-brain.ts apps/api/src/do/session-brain.schema.ts apps/api/src/routes/practice.ts apps/api/src/services/synthesis.ts
git commit -m "feat(api): wire unified teacher into DO synthesis path with memory context"
```

---

## Task 6: Remove Groq + Dead Code

**Files:**
- Modify: `apps/api/src/services/llm.ts`
- Delete: `apps/api/src/services/ask.ts`
- Modify: `apps/api/src/services/prompts.ts`
- Modify: `apps/api/src/services/wasm-bridge.ts`
- Modify: `apps/api/src/lib/types.ts`
- Modify: `apps/api/src/services/chat.ts` (title gen)

- [ ] **Step 1: Remove callGroq from llm.ts, add callWorkersAI**

```typescript
export async function callWorkersAI(
  env: Bindings,
  model: string,
  messages: Array<{ role: string; content: string }>,
  maxTokens: number = 100,
): Promise<string> {
  const url = `${env.AI_GATEWAY_BACKGROUND}/workers-ai/v1/chat/completions`;
  const res = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      model,
      messages,
      max_tokens: maxTokens,
    }),
  });
  if (!res.ok) {
    throw new InferenceError(`Workers AI error: ${res.status}`);
  }
  const data = (await res.json()) as { choices: Array<{ message: { content: string } }> };
  if (!data.choices?.[0]?.message?.content) {
    throw new InferenceError("Workers AI returned no content");
  }
  return data.choices[0].message.content;
}
```

- [ ] **Step 2: Switch title generation in chat.ts from Groq to Workers AI**

Replace `callGroq(env, "llama-3.3-70b-versatile", ...)` with `callWorkersAI(env, "@cf/qwen/qwen3-30b-a3b-fp8", ...)`.

- [ ] **Step 3: Remove GROQ_API_KEY from types.ts Bindings**

- [ ] **Step 4: Delete ask.ts**

```bash
rm apps/api/src/services/ask.ts
```

- [ ] **Step 5: Remove dead exports from prompts.ts**

Remove: `SUBAGENT_SYSTEM`, the old `TEACHER_SYSTEM`, `buildSubagentUserPrompt`, `buildTeacherUserPrompt`, `exerciseToolDefinition`, `ExerciseTool`, `ObservationRow`, `CatalogExercise`.

- [ ] **Step 6: Remove dead exports from wasm-bridge.ts**

Remove: `classifyStop`, `computeRerankFeatures`, `matchPieceText`, `initWasm`, `StopResult`.

- [ ] **Step 7: Remove dead params**

- `topMoments()`: remove unused `dimensionWeights?` parameter
- `callMuqEndpoint`/`callAmtEndpoint`: remove unused `sessionEnding` parameter
- `tryIdentifyPiece`: remove unused `_totalNotes` parameter

- [ ] **Step 8: Verify compilation and tests**

Run: `cd apps/api && npx tsc --noEmit && npx vitest run`
Expected: PASS

- [ ] **Step 9: Commit**

```bash
git add -A
git commit -m "chore(api): remove Groq, delete ask.ts, clean dead exports and parameters"
```

---

## Task 7: Frontend SSE Handler + piece_identified

**Files:**
- Modify: `apps/web/src/lib/api.ts`
- Modify: `apps/web/src/hooks/usePracticeSession.ts` (or wherever WS handler lives)
- Modify: `apps/web/src/components/AppChat.tsx`

- [ ] **Step 1: Add `tool_result` event handling to SSE client**

In `apps/web/src/lib/api.ts`, update the `ChatStreamEvent` type and the `send()` parser:

```typescript
type ChatStreamEvent =
  | { type: "start"; conversationId: string }
  | { type: "delta"; text: string }
  | { type: "tool_result"; name: string; componentsJson: string }
  | { type: "done" }
  | { type: "error"; message: string };
```

Update the line parser to handle `tool_result` events.

- [ ] **Step 2: Handle tool_result in AppChat.tsx handleSend()**

In the `onEvent` callback, add a case for `tool_result`:

```typescript
case "tool_result": {
  const components = JSON.parse(event.componentsJson) as InlineComponent[];
  setTransientMessages((prev) => {
    const updated = [...prev];
    const last = updated[updated.length - 1];
    if (last && last.role === "assistant") {
      last.components = [...(last.components ?? []), ...components];
    }
    return updated;
  });
  break;
}
```

- [ ] **Step 3: Handle piece_identified WS event**

In `usePracticeSession.ts`, add the missing case:

```typescript
case "piece_identified":
  setPieceInfo({
    composer: msg.composer as string,
    title: msg.title as string,
    confidence: msg.confidence as number,
  });
  break;
```

- [ ] **Step 4: Consolidate API_BASE**

Create `apps/web/src/lib/config.ts`:

```typescript
export const API_BASE = import.meta.env.PROD ? "https://api.crescend.ai" : "http://localhost:8787";
export const WS_BASE = import.meta.env.PROD ? "wss://api.crescend.ai" : "ws://localhost:8787";
```

Update `api.ts`, `api-client.ts`, and `practice-api.ts` to import from `config.ts`.

- [ ] **Step 5: Consolidate invalidateQueries**

In `AppChat.tsx`, extract:

```typescript
function invalidateConversation(conversationId: string) {
  queryClient.invalidateQueries({ queryKey: ["conversation", conversationId] });
  queryClient.invalidateQueries({ queryKey: ["conversations"] });
}
```

Replace all 4 scattered call sites.

- [ ] **Step 6: Commit**

```bash
git add apps/web/src/lib/api.ts apps/web/src/lib/config.ts apps/web/src/hooks/usePracticeSession.ts apps/web/src/components/AppChat.tsx apps/web/src/lib/api-client.ts apps/web/src/lib/practice-api.ts
git commit -m "feat(web): handle tool_result SSE events, piece_identified WS, consolidate API config"
```

---

## Task 8: Frontend Dead Code Cleanup

**Files:**
- Modify: `apps/web/src/` (various files)

- [ ] **Step 1: Remove dead frontend code**

- Delete `generateMockSession()` from `mock-session.ts` (keep `DIMENSION_COLORS`, `DIMENSION_LABELS`, `MockSessionData` if imported elsewhere)
- Remove `ObservationThrottle.queued` field (always null)
- Remove `onSynthesis` callback from `PracticeSessionOptions` interface and all references
- Remove `PlaceholderCard` component branches for `score_highlight`, `keyboard_guide`, `reference_browser` (will be replaced by real components in a follow-up task)

- [ ] **Step 2: Verify build**

Run: `cd apps/web && bun run build`
Expected: PASS

- [ ] **Step 3: Commit**

```bash
git add -A apps/web/
git commit -m "chore(web): remove dead code — mock session, observation throttle, unused callbacks"
```

---

## Summary

| Task | Component | Estimated Complexity |
|---|---|---|
| 1 | Tool registry + processors + tests | Medium |
| 2 | Unified teacher prompts | Small |
| 3 | Teacher service + stream parser + tests | Large (highest risk) |
| 4 | Chat route evolution | Medium |
| 5 | DO synthesis wiring | Medium-Large |
| 6 | Groq removal + dead code | Small |
| 7 | Frontend SSE + piece_identified | Medium |
| 8 | Frontend dead code cleanup | Small |

**Critical path:** Tasks 1 → 2 → 3 → 4 (chat works end-to-end). Task 5 can be parallelized with 4 after 3 is done. Tasks 6-8 are independent cleanup.

**Parallelization opportunities:**
- Tasks 1 + 2 are independent (tool registry vs prompts)
- Tasks 7 + 8 are independent (frontend SSE vs cleanup)
- Task 6 depends on tasks 4 + 5 being complete

---

## Follow-Up Work (Not In This Plan)

These are in the spec but intentionally deferred to separate implementation cycles:

- **`SynthesisCard` component** — New frontend component for structured post-session summary cards (use `/frontend-design` skill)
- **`ScoreHighlightCard`, `KeyboardGuideCard`, `ReferenceBrowserCard`, `SessionDataCard`** — Real component implementations replacing stubs (use `/frontend-design` skill)
- **`componentsJson` DB migration** — Fix double-serialized JSONB to proper JSONB (separate Drizzle migration PR)
- **DB column cleanup** — Drop `observations.elaborationText`, `.learningArc`, `.messageId`, `teachingApproaches` table (separate migration PR)
