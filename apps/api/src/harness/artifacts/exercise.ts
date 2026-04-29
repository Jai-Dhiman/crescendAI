import { z } from "zod";
import { DIMENSIONS } from "./diagnosis";

export const EXERCISE_TYPES = [
	"slow_practice",
	"isolated_hands",
	"rhythmic_variation",
	"segment_loop",
	"dynamic_exaggeration",
	"pedal_isolation",
] as const;

export const ACTION_REQUIRED_TYPES = [
	"segment_loop",
	"isolated_hands",
	"pedal_isolation",
] as const;

const ExerciseTypeEnum = z.enum(EXERCISE_TYPES);
const DimensionEnum = z.enum(DIMENSIONS);

// V5: action_binding inner shape is deferred to V6 tool registry
const ToolCallSpec = z.unknown();

export const ExerciseArtifactSchema = z
	.object({
		diagnosis_ref: z.string().min(1),
		diagnosis_summary: z.string().min(1).max(200),
		target_dimension: DimensionEnum,
		exercise_type: ExerciseTypeEnum,
		exercise_subtype: z.string().min(1).nullable(),
		bar_range: z.tuple([
			z.number().int().positive(),
			z.number().int().positive(),
		]),
		instruction: z.string().min(1).max(400),
		success_criterion: z.string().min(1).max(200),
		estimated_minutes: z.number().int().min(1).max(15),
		action_binding: ToolCallSpec.nullable(),
	})
	.refine(
		(e) =>
			!ACTION_REQUIRED_TYPES.includes(
				e.exercise_type as (typeof ACTION_REQUIRED_TYPES)[number],
			) || e.action_binding !== null,
		{
			message:
				"action_binding is required for exercise_type in {segment_loop, isolated_hands, pedal_isolation}",
			path: ["action_binding"],
		},
	)
	.refine((e) => e.bar_range[0] <= e.bar_range[1], {
		message: "bar_range start must be <= end",
		path: ["bar_range"],
	});

export type ExerciseArtifact = z.infer<typeof ExerciseArtifactSchema>;
