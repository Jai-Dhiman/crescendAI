import { eq, sql } from "drizzle-orm";
import type { Db, Bindings } from "../lib/types";
import type { SessionAccumulator } from "./accumulator";
import { callAnthropic } from "./llm";
import { SESSION_SYNTHESIS_SYSTEM } from "./prompts";
import { messages } from "../db/schema/conversations";
import { observations } from "../db/schema/observations";
import { sessions } from "../db/schema/sessions";
import type { Dimension } from "../lib/dims";
import { DIMS_6 } from "../lib/dims";

// SCALER_MEAN defaults from the Rust stop classifier (same values)
const SCALER_MEAN: Record<Dimension, number> = {
  dynamics: 0.5450,
  timing: 0.4848,
  pedaling: 0.4594,
  articulation: 0.5369,
  phrasing: 0.5188,
  interpretation: 0.5064,
};

export interface SynthesisContext {
  sessionId: string;
  studentId: string;
  conversationId: string | null;
  baselines: Record<Dimension, number> | null;
  pieceContext: { composer: string; title: string; pieceId: string } | null;
  studentMemory: string | null;
  totalChunks: number;
  sessionDurationMs: number;
}

export interface SynthesisResult {
  text: string;
  isFallback: boolean;
}

function buildPracticePattern(
  acc: SessionAccumulator,
  sessionDurationMs: number,
): unknown[] {
  if (acc.modeTransitions.length === 0) {
    return [];
  }

  const entries: unknown[] = [];

  for (let i = 0; i < acc.modeTransitions.length; i++) {
    const tr = acc.modeTransitions[i];
    const endTs =
      i + 1 < acc.modeTransitions.length
        ? acc.modeTransitions[i + 1].timestampMs
        : sessionDurationMs;

    const durationMin =
      Math.round(
        (Math.max(0, endTs - tr.timestampMs) / 60_000) * 10,
      ) / 10;

    const entry: Record<string, unknown> = {
      mode: tr.to.toLowerCase(),
      duration_min: durationMin,
    };

    if (tr.to.toLowerCase() === "drilling") {
      const dr = acc.drillingRecords.find(
        (r) => r.startedAtChunk === tr.chunkIndex,
      );
      if (dr) {
        if (dr.barRange) {
          entry["bar_range"] = dr.barRange;
        }
        entry["repetitions"] = dr.repetitionCount;
      }
    }

    entries.push(entry);
  }

  return entries;
}

function momentToJson(m: SessionAccumulator["teachingMoments"][number]): unknown {
  const deviationRounded = Math.round(m.deviation * 1000) / 1000;
  const obj: Record<string, unknown> = {
    dimension: m.dimension,
    deviation: deviationRounded,
    is_positive: m.isPositive,
    reasoning: m.reasoning,
  };
  if (m.barRange) {
    obj["bar_range"] = m.barRange;
  }
  return obj;
}

export function buildSynthesisPrompt(
  acc: SessionAccumulator,
  context: SynthesisContext,
): string {
  const durationMin =
    Math.round((context.sessionDurationMs / 60_000) * 10) / 10;

  const practicePattern = buildPracticePattern(acc, context.sessionDurationMs);
  const topMoments = acc.topMoments().map(momentToJson);

  const obj: Record<string, unknown> = {
    session_duration_minutes: durationMin,
    chunks_processed: context.totalChunks,
    practice_pattern: practicePattern,
    top_moments: topMoments,
  };

  if (context.baselines !== null) {
    obj["baselines"] = {
      dynamics: context.baselines.dynamics,
      timing: context.baselines.timing,
      pedaling: context.baselines.pedaling,
      articulation: context.baselines.articulation,
      phrasing: context.baselines.phrasing,
      interpretation: context.baselines.interpretation,
    };
  } else {
    obj["baselines"] = null;
  }

  if (context.pieceContext !== null) {
    obj["piece"] = {
      composer: context.pieceContext.composer,
      title: context.pieceContext.title,
    };
  }

  if (context.studentMemory !== null) {
    obj["student_memory"] = context.studentMemory;
  }

  if (acc.drillingRecords.length > 0) {
    const dims = DIMS_6;
    const drillingProgress = acc.drillingRecords.map((dr) => {
      const firstScores: Record<string, number> = {};
      const finalScores: Record<string, number> = {};
      for (let i = 0; i < dims.length; i++) {
        firstScores[dims[i]] = Math.round((dr.firstScores[i] ?? 0) * 1000) / 1000;
        finalScores[dims[i]] = Math.round((dr.finalScores[i] ?? 0) * 1000) / 1000;
      }
      const entry: Record<string, unknown> = {
        repetitions: dr.repetitionCount,
        first_scores: firstScores,
        final_scores: finalScores,
      };
      if (dr.barRange) {
        entry["bar_range"] = dr.barRange;
      }
      return entry;
    });
    obj["drilling_progress"] = drillingProgress;
  }

  return JSON.stringify(obj, null, 2);
}

export async function callSynthesisLlm(
  env: Bindings,
  promptContext: string,
): Promise<SynthesisResult> {
  const fallback: SynthesisResult = {
    text: "I had trouble preparing your feedback. Try playing again and I'll have more to say next time.",
    isFallback: true,
  };

  try {
    const response = await callAnthropic(env, {
      model: "claude-sonnet-4-20250514",
      max_tokens: 1024,
      system: [
        {
          type: "text",
          text: SESSION_SYNTHESIS_SYSTEM,
          cache_control: { type: "ephemeral" },
        },
      ],
      messages: [{ role: "user", content: promptContext }],
    });

    const firstContent = response.content[0];
    if (!firstContent || firstContent.type !== "text" || !firstContent.text) {
      console.error(
        JSON.stringify({
          level: "error",
          message: "synthesis LLM returned no text content",
          stopReason: response.stop_reason,
        }),
      );
      return fallback;
    }

    return { text: firstContent.text, isFallback: false };
  } catch (err) {
    const error = err as Error;
    console.error(
      JSON.stringify({
        level: "error",
        message: "synthesis LLM call failed",
        error: error.message,
        stack: error.stack,
      }),
    );
    return fallback;
  }
}

export async function persistSynthesisMessage(
  db: Db,
  conversationId: string,
  text: string,
  sessionId: string,
): Promise<void> {
  await db.insert(messages).values({
    conversationId,
    role: "assistant",
    content: text,
    messageType: "synthesis",
    sessionId,
  });
}

export async function persistAccumulatedMoments(
  db: Db,
  studentId: string,
  sessionId: string,
  conversationId: string | null,
  moments: SessionAccumulator["teachingMoments"],
): Promise<void> {
  for (const moment of moments) {
    const framing = moment.isPositive ? "recognition" : "correction";
    const reasoningTrace = moment.llmAnalysis ?? moment.reasoning;

    await db
      .insert(observations)
      .values({
        studentId,
        sessionId,
        chunkIndex: moment.chunkIndex,
        dimension: moment.dimension,
        observationText: moment.reasoning,
        reasoningTrace,
        framing,
        dimensionScore: moment.score,
        studentBaseline: moment.baseline,
        isFallback: false,
        conversationId,
      })
      .onConflictDoNothing();
  }
}

export async function clearNeedsSynthesis(
  db: Db,
  sessionId: string,
): Promise<void> {
  await db
    .update(sessions)
    .set({ needsSynthesis: false })
    .where(eq(sessions.id, sessionId));
}

export async function loadBaselinesFromDb(
  db: Db,
  studentId: string,
): Promise<Record<Dimension, number> | null> {
  const thirtyDaysAgo = new Date(Date.now() - 30 * 24 * 60 * 60 * 1000);

  const rows = await db
    .select({
      dimension: observations.dimension,
      avgScore: sql<number>`AVG(${observations.dimensionScore})`,
    })
    .from(observations)
    .where(
      sql`${observations.studentId} = ${studentId} AND ${observations.createdAt} > ${thirtyDaysAgo}`,
    )
    .groupBy(observations.dimension);

  if (rows.length === 0) {
    return null;
  }

  const dimMap: Partial<Record<Dimension, number>> = {};
  for (const row of rows) {
    const dim = row.dimension as Dimension;
    if (DIMS_6.includes(dim) && row.avgScore !== null) {
      dimMap[dim] = row.avgScore;
    }
  }

  const result = {} as Record<Dimension, number>;
  for (const dim of DIMS_6) {
    result[dim] = dimMap[dim] ?? SCALER_MEAN[dim];
  }

  return result;
}
