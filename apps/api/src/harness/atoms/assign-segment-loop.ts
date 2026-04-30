import { z } from "zod";
import { DIMS_6 } from "../../lib/dims";
import { ToolPreconditionError, ValidationError } from "../../lib/errors";
import { createSegmentLoop } from "../../services/segment-loops";
import type { SegmentLoopArtifact } from "../artifacts/segment-loop";
import type { PhaseContext, ToolDefinition } from "../loop/types";
import { createDb } from "../../db/client";

const inputSchema = z.object({
  piece_id: z.string().min(1).optional(),
  bars_start: z.number().int().positive(),
  bars_end: z.number().int().positive(),
  required_correct: z.number().int().min(1).max(10).default(5),
  dimension: z.enum([...DIMS_6]).optional().nullable(),
});

export async function assignSegmentLoopAtom(
  ctx: PhaseContext,
  rawInput: unknown,
): Promise<SegmentLoopArtifact> {
  const parsed = inputSchema.safeParse(rawInput);
  if (!parsed.success) {
    throw new ValidationError(parsed.error.message);
  }
  const input = parsed.data;

  if (!input.piece_id) {
    throw new ToolPreconditionError("no_piece_identified");
  }

  if (input.bars_end < input.bars_start) {
    throw new ValidationError("bars_end must be >= bars_start");
  }

  if (!ctx.trigger) {
    throw new ToolPreconditionError("assign_segment_loop requires trigger on PhaseContext");
  }

  const db = createDb(ctx.env.HYPERDRIVE);
  return createSegmentLoop(db, {
    studentId: ctx.studentId,
    pieceId: input.piece_id,
    conversationId: ctx.conversationId,
    barsStart: input.bars_start,
    barsEnd: input.bars_end,
    requiredCorrect: input.required_correct,
    dimension: input.dimension ?? null,
    trigger: ctx.trigger,
  });
}

export const ASSIGN_SEGMENT_LOOP_TOOL: ToolDefinition = {
  name: "assign_segment_loop",
  description:
    "Assign a focused passage-loop practice task. The student will practice the specified bar range repeatedly until they complete the required number of isolated attempts. Use after identifying a specific passage that needs targeted work. Requires a piece to be identified first.",
  input_schema: {
    type: "object",
    properties: {
      piece_id: {
        type: "string",
        description:
          "Piece slug from search_catalog (e.g. 'chopin.ballades.1'). Pass verbatim.",
      },
      bars_start: {
        type: "integer",
        description: "First bar of the practice passage (inclusive, 1-indexed).",
        minimum: 1,
      },
      bars_end: {
        type: "integer",
        description: "Last bar of the practice passage (inclusive). Must be >= bars_start.",
        minimum: 1,
      },
      required_correct: {
        type: "integer",
        description:
          "Number of isolated loop attempts required to complete the assignment (1-10). Default 5.",
        minimum: 1,
        maximum: 10,
        default: 5,
      },
      dimension: {
        type: "string",
        enum: [...DIMS_6],
        description: "Optional: which musical dimension this loop targets.",
      },
    },
    required: ["bars_start", "bars_end"],
  },
  invoke: async (input: unknown, ctx?: PhaseContext): Promise<unknown> => {
    if (!ctx) throw new ToolPreconditionError("assign_segment_loop requires PhaseContext");
    return assignSegmentLoopAtom(ctx, input);
  },
};
