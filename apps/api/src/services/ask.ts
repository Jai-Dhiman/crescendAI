import { eq, and } from "drizzle-orm";
import type { ServiceContext, Bindings } from "../lib/types";
import { callGroq, callAnthropic } from "./llm";
import { buildMemoryContext } from "./memory";
import { exercises, exerciseDimensions } from "../db/schema/exercises";
import { observations, teachingApproaches } from "../db/schema/observations";
import { DIMS_6 } from "../lib/dims";
import {
  SUBAGENT_SYSTEM,
  TEACHER_SYSTEM,
  exerciseToolDefinition as sharedExerciseToolDef,
} from "./prompts";
import { InferenceError, ValidationError } from "../lib/errors";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export interface AskRequest {
  teachingMoment: unknown;
  studentId: string;
  sessionId: string;
  pieceContext?: unknown;
}

export interface AskResponse {
  observationText: string;
  dimension: string;
  framing: string;
  reasoningTrace: string;
  isFallback: boolean;
  componentsJson: string | null;
}

interface CatalogExercise {
  id: string;
  title: string;
  description: string;
  difficulty: string;
}

interface SubagentJson {
  selected_moment?: {
    chunk_index?: number;
    dimension?: string;
    dimension_score?: number;
    student_baseline?: number;
    bar_range?: string | null;
    section_label?: string | null;
  };
  framing?: string;
  learning_arc?: string;
  is_positive?: boolean;
  musical_context?: string;
}


// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function dimensionDescription(dimension: string): string {
  switch (dimension) {
    case "dynamics":
      return "dynamic range and volume control";
    case "timing":
      return "rhythmic accuracy and tempo consistency";
    case "pedaling":
      return "pedal clarity and harmonic changes";
    case "articulation":
      return "note clarity and touch";
    case "phrasing":
      return "musical phrasing and melodic shaping";
    case "interpretation":
      return "musical expression and stylistic choices";
    default:
      return "that aspect of your playing";
  }
}

function fallbackObservation(dimension: string): string {
  return (
    `I noticed your ${dimension} could use some attention in that last section. ` +
    `Try recording yourself and listening back -- sometimes it's hard to hear ` +
    `${dimensionDescription(dimension)} while you're playing.`
  );
}

/**
 * Extract a JSON block from text that may contain markdown code fences.
 * Returns the raw JSON string (not parsed) or null.
 */
function extractJsonBlock(text: string): string | null {
  // Try ```json ... ``` fences first
  const fencedStart = text.indexOf("```json");
  if (fencedStart !== -1) {
    const jsonStart = fencedStart + 7;
    const fencedEnd = text.indexOf("```", jsonStart);
    if (fencedEnd !== -1) {
      return text.slice(jsonStart, fencedEnd).trim();
    }
  }

  // Try ``` ... ``` with a leading brace
  const rawFence = text.indexOf("```\n{");
  if (rawFence !== -1) {
    const jsonStart = rawFence + 4;
    const fencedEnd = text.indexOf("```", jsonStart);
    if (fencedEnd !== -1) {
      return text.slice(jsonStart, fencedEnd).trim();
    }
  }

  // Fallback: find the outermost { ... }
  const objStart = text.indexOf("{");
  const objEnd = text.lastIndexOf("}");
  if (objStart !== -1 && objEnd !== -1 && objEnd > objStart) {
    const candidate = text.slice(objStart, objEnd + 1);
    try {
      JSON.parse(candidate);
      return candidate;
    } catch {
      return null;
    }
  }

  return null;
}

/**
 * Split subagent output into the JSON block (as a string) and the trailing narrative.
 */
export function splitSubagentOutput(text: string): {
  json: SubagentJson | null;
  narrative: string;
} {
  const jsonStr = extractJsonBlock(text);
  if (!jsonStr) {
    return { json: null, narrative: text.trim() };
  }

  let parsed: SubagentJson;
  try {
    parsed = JSON.parse(jsonStr) as SubagentJson;
  } catch {
    return { json: null, narrative: text.trim() };
  }

  // Everything after the closing brace of the JSON block is the narrative
  const braceEnd = text.lastIndexOf("}");
  let narrativeRaw = text.slice(braceEnd + 1);

  // Skip past ``` if there was a code fence
  const fenceClose = narrativeRaw.indexOf("```");
  if (fenceClose !== -1) {
    narrativeRaw = narrativeRaw.slice(fenceClose + 3);
  }

  const narrative = narrativeRaw.trim();
  return { json: parsed, narrative };
}

/**
 * Strip markdown bold markers, remove surrounding quotes, truncate at 500 chars on
 * the last period.
 */
export function postProcessObservation(text: string): string {
  let cleaned = text.trim();

  // Strip surrounding quotes
  if (cleaned.startsWith('"') && cleaned.endsWith('"')) {
    cleaned = cleaned.slice(1, -1);
  }

  // Strip markdown bold/italic markers
  cleaned = cleaned.replaceAll("**", "").replaceAll("__", "");

  if (cleaned.length > 500) {
    const slice = cleaned.slice(0, 500);
    const lastPeriod = slice.lastIndexOf(".");
    if (lastPeriod !== -1) {
      cleaned = cleaned.slice(0, lastPeriod + 1);
    } else {
      cleaned = slice;
    }
  }

  return cleaned;
}

// ---------------------------------------------------------------------------
// Prompt builders (ported from Rust prompts.rs)
// ---------------------------------------------------------------------------

function buildSubagentUserPrompt(
  teachingMoment: unknown,
  memoryContext: string,
  recentObs: Array<{ dimension: string; observationText: string; framing: string | null }>,
  pieceContext?: unknown,
): string {
  const tm = teachingMoment as Record<string, unknown>;
  const pc = pieceContext as Record<string, unknown> | undefined;

  const parts: string[] = [];

  // Teaching moment block
  parts.push("<teaching_moment>");
  parts.push(
    `Chunk ${tm?.chunk_index ?? 0} at ${tm?.start_offset_sec ?? 0}s into session.`,
  );
  parts.push(
    `Dimension flagged: ${tm?.dimension ?? "unknown"} (score: ${((tm?.dimension_score as number) ?? 0).toFixed(2)}, stop probability: ${((tm?.stop_probability as number) ?? 0).toFixed(2)})`,
  );

  const allScores = tm?.all_scores as Record<string, number> | undefined;
  if (allScores) {
    parts.push("All 6 dimension scores for this chunk:");
    for (const dim of DIMS_6) {
      if (allScores[dim] != null) {
        parts.push(`- ${dim}: ${(allScores[dim] as number).toFixed(2)}`);
      }
    }
  }
  parts.push("</teaching_moment>");
  parts.push("");

  // Piece context block
  if (pc) {
    parts.push("<piece_context>");
    if (pc.composer) parts.push(`Composer: ${pc.composer}`);
    if (pc.title) parts.push(`Title: ${pc.title}`);
    if (pc.bar_range) parts.push(`Bar range: ${pc.bar_range}`);
    if (pc.analysis_tier != null)
      parts.push(
        `Analysis tier: ${pc.analysis_tier} (1=full score context, 2=absolute, 3=scores only)`,
      );

    const musicalAnalysis = pc.musical_analysis as Array<Record<string, string>> | undefined;
    if (musicalAnalysis?.length) {
      parts.push("");
      parts.push("<musical_analysis>");
      for (const dimAnalysis of musicalAnalysis) {
        const dim = dimAnalysis.dimension;
        if (!dim) continue;
        parts.push(`<${dim}>`);
        if (dimAnalysis.analysis) parts.push(`  ${dimAnalysis.analysis}`);
        if (dimAnalysis.score_marking) parts.push(`  Score marking: ${dimAnalysis.score_marking}`);
        if (dimAnalysis.reference_comparison)
          parts.push(`  Reference: ${dimAnalysis.reference_comparison}`);
        parts.push(`</${dim}>`);
      }
      parts.push("</musical_analysis>");
    }
    parts.push("</piece_context>");
    parts.push("");
  }

  // Student context (simplified -- level inferred from context)
  parts.push("<student_context>");
  parts.push("Level: intermediate");
  parts.push("</student_context>");
  parts.push("");

  // Memory context
  if (memoryContext) {
    parts.push("<memory>");
    parts.push(memoryContext);
    parts.push("</memory>");
    parts.push("");
  }

  // Recent observations
  if (recentObs.length > 0) {
    parts.push("<recent_observations>");
    for (const obs of recentObs) {
      parts.push(
        `- [${obs.dimension}] "${obs.observationText}" (framing: ${obs.framing ?? "unknown"})`,
      );
    }
    parts.push("</recent_observations>");
    parts.push("");
  }

  parts.push(
    "<task>\nAnalyze the teaching moment above. Select the best observation to make and decide how to frame it. Output the JSON + narrative as specified.\n</task>",
  );

  return parts.join("\n");
}

function buildTeacherUserPrompt(
  subagentJsonStr: string,
  narrative: string,
  catalog: CatalogExercise[],
): string {
  const parts: string[] = [];

  parts.push("<analysis>");
  parts.push(subagentJsonStr);
  parts.push("");
  parts.push(narrative);
  parts.push("</analysis>");
  parts.push("");

  parts.push("<student>");
  parts.push("Level: intermediate");
  parts.push("</student>");
  parts.push("");

  if (catalog.length > 0) {
    parts.push("<available_exercises>");
    parts.push(
      "These curated exercises are available. If one fits, you can reference it by ID in your exercise tool call.",
    );
    for (const ex of catalog) {
      parts.push(`- [${ex.id}] ${ex.title} (${ex.difficulty}): ${ex.description}`);
    }
    parts.push("</available_exercises>");
    parts.push("");
  }

  if (catalog.length > 0) {
    parts.push(
      "<task>\nBased on the analysis above, give one observation to the student. Be specific about what you heard and what to try. 1-3 sentences, no formatting.\n\nIf the student would benefit from a concrete practice drill, use the create_exercise tool to attach one. Most observations should be text-only -- only create an exercise when structured practice would genuinely help more than verbal guidance.\n</task>",
    );
  } else {
    parts.push(
      "<task>\nBased on the analysis above, give one observation to the student. Be specific about what you heard and what to try. 1-3 sentences, no formatting.\n</task>",
    );
  }

  return parts.join("\n");
}

// ---------------------------------------------------------------------------
// DB helpers
// ---------------------------------------------------------------------------

async function lookupCatalogExercises(
  ctx: ServiceContext,
  dimension: string,
): Promise<CatalogExercise[]> {
  const rows = await ctx.db
    .select({
      id: exercises.id,
      title: exercises.title,
      description: exercises.description,
      difficulty: exercises.difficulty,
    })
    .from(exercises)
    .innerJoin(
      exerciseDimensions,
      eq(exerciseDimensions.exerciseId, exercises.id),
    )
    .where(
      and(
        eq(exerciseDimensions.dimension, dimension),
        eq(exercises.source, "curated"),
      ),
    )
    .limit(5);

  return rows;
}

/**
 * Persist a single teacher-generated exercise and link its dimension.
 * Returns the new exercise UUID.
 */
async function persistGeneratedExercise(
  ctx: ServiceContext,
  title: string,
  instruction: string,
  focusDimension: string,
  sourcePassage: string,
  targetSkill: string,
): Promise<string> {
  const [inserted] = await ctx.db
    .insert(exercises)
    .values({
      title,
      description: `${targetSkill} -- ${sourcePassage}`,
      instructions: instruction,
      difficulty: "intermediate",
      category: "generated",
      source: "teacher_llm",
    })
    .returning({ id: exercises.id });

  if (!inserted) {
    throw new InferenceError("Failed to insert generated exercise");
  }

  await ctx.db.insert(exerciseDimensions).values({
    exerciseId: inserted.id,
    dimension: focusDimension,
  });

  return inserted.id;
}

/**
 * Validate a `create_exercise` tool input, persist each exercise, and return
 * the components JSON string ready for the response.
 */
export async function processExerciseToolCall(
  ctx: ServiceContext,
  toolInput: unknown,
  _dimension: string,
): Promise<string | null> {
  const input = toolInput as Record<string, unknown>;

  const sourcePassage = input.source_passage;
  const targetSkill = input.target_skill;
  const exercisesArr = input.exercises;

  if (typeof sourcePassage !== "string" || !sourcePassage) {
    throw new ValidationError("Missing source_passage in exercise tool call");
  }
  if (typeof targetSkill !== "string" || !targetSkill) {
    throw new ValidationError("Missing target_skill in exercise tool call");
  }
  if (!Array.isArray(exercisesArr) || exercisesArr.length === 0) {
    throw new ValidationError("Missing or empty exercises array in exercise tool call");
  }

  const processed: unknown[] = [];

  for (const ex of exercisesArr as Array<Record<string, unknown>>) {
    const title = typeof ex.title === "string" ? ex.title : "Practice Drill";
    const instruction = typeof ex.instruction === "string" ? ex.instruction : "";
    const rawDim = typeof ex.focus_dimension === "string" ? ex.focus_dimension : "dynamics";
    const focusDim = (DIMS_6 as readonly string[]).includes(rawDim) ? rawDim : "dynamics";
    const hands = typeof ex.hands === "string" ? ex.hands : undefined;

    let exerciseId: string;

    // If the teacher referenced a catalog exercise by ID, use it directly
    if (typeof ex.exercise_id === "string" && ex.exercise_id) {
      exerciseId = ex.exercise_id;
    } else {
      exerciseId = await persistGeneratedExercise(
        ctx,
        title,
        instruction,
        focusDim,
        sourcePassage,
        targetSkill,
      );
    }

    const exJson: Record<string, unknown> = {
      title,
      instruction,
      focusDimension: focusDim,
      exerciseId,
    };
    if (hands) {
      exJson.hands = hands;
    }
    processed.push(exJson);
  }

  const component = [
    {
      type: "exercise_set",
      config: {
        sourcePassage,
        targetSkill,
        exercises: processed,
      },
    },
  ];

  return JSON.stringify(component);
}

// ---------------------------------------------------------------------------
// Core pipeline
// ---------------------------------------------------------------------------

/**
 * Core two-stage LLM pipeline.
 * No D1 persistence for the observation itself (caller is responsible).
 * Called by both the HTTP handler and the SessionBrain DO.
 */
export async function handleAskInner(
  env: Bindings,
  ctx: ServiceContext,
  request: AskRequest,
): Promise<AskResponse> {
  const tm = request.teachingMoment as Record<string, unknown>;
  const dimension =
    typeof tm?.dimension === "string" ? tm.dimension : "unknown";

  // Build memory context
  const memoryContext = await buildMemoryContext(ctx, request.studentId);

  // Pull recent observations from memory context output (already formatted as text)
  // The memory service returns a formatted string; pass it directly to the subagent prompt.
  // We also need typed recent observations for the prompt template -- re-query them.
  const recentObs = await ctx.db
    .select({
      dimension: observations.dimension,
      observationText: observations.observationText,
      framing: observations.framing,
    })
    .from(observations)
    .where(eq(observations.studentId, request.studentId))
    .orderBy(observations.createdAt)
    .limit(5);

  // Stage 1: Subagent (Groq)
  const subagentUserPrompt = buildSubagentUserPrompt(
    request.teachingMoment,
    memoryContext,
    recentObs,
    request.pieceContext,
  );

  let subagentOutput: string;
  try {
    subagentOutput = await callGroq(
      env,
      "llama-3.3-70b-versatile",
      [
        { role: "system", content: SUBAGENT_SYSTEM },
        { role: "user", content: subagentUserPrompt },
      ],
      800,
    );
  } catch (err) {
    console.error(
      JSON.stringify({
        level: "error",
        message: "Subagent (Groq) failed",
        error: err instanceof Error ? err.message : String(err),
        studentId: request.studentId,
        sessionId: request.sessionId,
      }),
    );
    return {
      observationText: fallbackObservation(dimension),
      dimension,
      framing: "correction",
      reasoningTrace: "{}",
      isFallback: true,
      componentsJson: null,
    };
  }

  const { json: subagentJson, narrative: subagentNarrative } =
    splitSubagentOutput(subagentOutput);

  if (!subagentJson) {
    console.error(
      JSON.stringify({
        level: "error",
        message: "Subagent JSON parse failed",
        rawOutput: subagentOutput.slice(0, 500),
        studentId: request.studentId,
      }),
    );
    return {
      observationText: fallbackObservation(dimension),
      dimension,
      framing: "correction",
      reasoningTrace: JSON.stringify({ subagent_output: subagentOutput }),
      isFallback: true,
      componentsJson: null,
    };
  }

  const framing =
    typeof subagentJson.framing === "string" ? subagentJson.framing : "correction";

  // Stage 2: Teacher (Anthropic)
  const catalog = await lookupCatalogExercises(ctx, dimension);

  const teacherUserPrompt = buildTeacherUserPrompt(
    JSON.stringify(subagentJson, null, 2),
    subagentNarrative,
    catalog,
  );

  let observationText: string;
  let isFallback = false;
  let componentsJson: string | null = null;

  try {
    const teacherResponse = await callAnthropic(env, {
      model: "claude-sonnet-4-20250514",
      max_tokens: 500,
      system: TEACHER_SYSTEM,
      messages: [{ role: "user", content: teacherUserPrompt }],
      tools: [sharedExerciseToolDef()],
      tool_choice: { type: "auto" },
    });

    // Extract text from response content blocks
    const textBlock = teacherResponse.content.find((b) => b.type === "text");
    const rawText = textBlock?.text ?? fallbackObservation(dimension);
    observationText = postProcessObservation(rawText);

    // Process tool_use block if present
    const toolBlock = teacherResponse.content.find(
      (b) => b.type === "tool_use" && b.name === "create_exercise",
    );
    if (toolBlock?.input != null) {
      try {
        componentsJson = await processExerciseToolCall(ctx, toolBlock.input, dimension);
      } catch (toolErr) {
        console.error(
          JSON.stringify({
            level: "error",
            message: "Exercise tool call processing failed",
            error: toolErr instanceof Error ? toolErr.message : String(toolErr),
            studentId: request.studentId,
          }),
        );
        // Non-fatal: observation still delivered without exercise
      }
    }
  } catch (err) {
    console.error(
      JSON.stringify({
        level: "error",
        message: "Teacher LLM (Anthropic) failed",
        error: err instanceof Error ? err.message : String(err),
        studentId: request.studentId,
        sessionId: request.sessionId,
      }),
    );
    observationText = fallbackObservation(dimension);
    isFallback = true;
  }

  const reasoningTrace = JSON.stringify({ subagent_output: subagentOutput });

  return {
    observationText,
    dimension,
    framing,
    reasoningTrace,
    isFallback,
    componentsJson,
  };
}

// ---------------------------------------------------------------------------
// DB persistence helpers (called by HTTP handler, not core pipeline)
// ---------------------------------------------------------------------------

export async function storeObservation(
  ctx: ServiceContext,
  params: {
    id: string;
    studentId: string;
    sessionId: string;
    chunkIndex: number | null;
    dimension: string;
    observationText: string;
    reasoningTrace: string;
    framing: string;
    dimensionScore: number | null;
    studentBaseline: number | null;
    pieceContext: string | null;
    isFallback: boolean;
  },
): Promise<void> {
  await ctx.db.insert(observations).values({
    id: params.id,
    studentId: params.studentId,
    sessionId: params.sessionId,
    chunkIndex: params.chunkIndex ?? undefined,
    dimension: params.dimension,
    observationText: params.observationText,
    reasoningTrace: params.reasoningTrace,
    framing: params.framing,
    dimensionScore: params.dimensionScore ?? undefined,
    studentBaseline: params.studentBaseline ?? undefined,
    pieceContext: params.pieceContext ?? undefined,
    isFallback: params.isFallback,
  });
}

export async function storeTeachingApproach(
  ctx: ServiceContext,
  params: {
    id: string;
    studentId: string;
    observationId: string;
    dimension: string;
    framing: string;
    approachSummary: string;
  },
): Promise<void> {
  await ctx.db.insert(teachingApproaches).values({
    id: params.id,
    studentId: params.studentId,
    observationId: params.observationId,
    dimension: params.dimension,
    framing: params.framing,
    approachSummary: params.approachSummary,
  });
}
