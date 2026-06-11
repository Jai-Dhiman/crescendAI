import { z } from "zod";
import { DIMS_6 } from "../../lib/dims";

const DimensionEnum = z.enum(DIMS_6 as unknown as [string, ...string[]]);

const barRangeSchema = z
  .tuple([z.number().int().positive(), z.number().int().positive()])
  .refine(([start, end]) => start <= end, {
    message: "bar_range start must be <= end",
    path: ["bar_range"],
  });

const tempoFactorSchema = z.number().min(0.25).max(1.0);

export const OwnPassageLoopSchema = z.object({
  kind: z.literal("own_passage_loop"),
  target_dimension: DimensionEnum,
  bar_range: barRangeSchema,
  tempo_factor: tempoFactorSchema,
}).strict();

export const CorpusDrillSchema = z.object({
  kind: z.literal("corpus_drill"),
  target_dimension: DimensionEnum,
  bar_range: barRangeSchema,
  tempo_factor: tempoFactorSchema,
  primitive_id: z.string().nullable().optional().default(null),
}).strict();

export const ExerciseRoutingDecisionSchema = z.discriminatedUnion("kind", [
  OwnPassageLoopSchema,
  CorpusDrillSchema,
]);

export type OwnPassageLoopDecision = z.infer<typeof OwnPassageLoopSchema>;
export type CorpusDrillDecision = z.infer<typeof CorpusDrillSchema>;
export type ExerciseRoutingDecision = z.infer<typeof ExerciseRoutingDecisionSchema>;
