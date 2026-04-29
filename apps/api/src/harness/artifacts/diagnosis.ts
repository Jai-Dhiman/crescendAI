import { z } from "zod";

export const DIMENSIONS = [
	"dynamics",
	"timing",
	"pedaling",
	"articulation",
	"phrasing",
	"interpretation",
] as const;
export const SEVERITIES = ["minor", "moderate", "significant"] as const;
export const SCOPES = ["stop_moment", "passage", "session"] as const;
export const CONFIDENCES = ["low", "medium", "high"] as const;
export const FINDING_TYPES = ["issue", "strength", "neutral"] as const;

const DimensionEnum = z.enum(DIMENSIONS);
const SeverityEnum = z.enum(SEVERITIES);
const ScopeEnum = z.enum(SCOPES);
const ConfidenceEnum = z.enum(CONFIDENCES);

export const DiagnosisArtifactSchema = z
	.object({
		primary_dimension: DimensionEnum,
		dimensions: z.array(DimensionEnum).min(1),
		severity: SeverityEnum,
		scope: ScopeEnum,
		bar_range: z
			.tuple([z.number().int().positive(), z.number().int().positive()])
			.nullable(),
		evidence_refs: z.array(z.string().min(1)).min(1),
		one_sentence_finding: z.string().min(1).max(200),
		confidence: ConfidenceEnum,
		finding_type: z.enum(FINDING_TYPES),
	})
	.refine((d) => d.dimensions.includes(d.primary_dimension), {
		message: "primary_dimension must be included in dimensions",
		path: ["primary_dimension"],
	})
	.refine((d) => d.scope === "session" || d.bar_range !== null, {
		message: 'bar_range may be null only when scope is "session"',
		path: ["bar_range"],
	})
	.refine((d) => d.bar_range === null || d.bar_range[0] <= d.bar_range[1], {
		message: "bar_range start must be <= end",
		path: ["bar_range"],
	});

export type DiagnosisArtifact = z.infer<typeof DiagnosisArtifactSchema>;
