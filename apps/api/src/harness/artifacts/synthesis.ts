import { z } from "zod";
import { DIMENSIONS, SEVERITIES } from "./diagnosis";

export const SYNTHESIS_SCOPES = [
	"session",
	"weekly",
	"piece_onboarding",
] as const;

const DimensionEnum = z.enum(DIMENSIONS);
const SeverityEnum = z.enum(SEVERITIES);
const SynthesisScopeEnum = z.enum(SYNTHESIS_SCOPES);

const StrengthEntry = z.object({
	dimension: DimensionEnum,
	one_liner: z.string().min(1).max(200),
});

const FocusAreaEntry = z.object({
	dimension: DimensionEnum,
	one_liner: z.string().min(1).max(200),
	severity: SeverityEnum,
});

export const SynthesisArtifactSchema = z
	.object({
		session_id: z.string().min(1),
		synthesis_scope: SynthesisScopeEnum,
		strengths: z.array(StrengthEntry).max(2),
		focus_areas: z.array(FocusAreaEntry).max(3),
		proposed_exercises: z.array(z.string().min(1)).max(3),
		dominant_dimension: DimensionEnum,
		recurring_pattern: z.string().min(1).nullable(),
		next_session_focus: z.string().min(1).max(200).nullable(),
		diagnosis_refs: z.array(z.string().min(1)),
		headline: z.string().min(300).max(500),
	})
	.refine(
		(s) => s.synthesis_scope !== "weekly" || s.recurring_pattern !== null,
		{
			message: 'recurring_pattern is required when synthesis_scope is "weekly"',
			path: ["recurring_pattern"],
		},
	)
	.refine(
		(s) =>
			s.synthesis_scope !== "piece_onboarding" ||
			s.focus_areas.every((f) => f.severity === "minor"),
		{
			message:
				'when synthesis_scope is "piece_onboarding", all focus_areas[].severity must be "minor"',
			path: ["focus_areas"],
		},
	);

export type SynthesisArtifact = z.infer<typeof SynthesisArtifactSchema>;
