import { z } from "zod";
import { entityRefSchema } from "../entities";
import { evidenceRefSchema } from "../content";

const SIX_DIM = [
	"dynamics",
	"timing",
	"pedaling",
	"articulation",
	"phrasing",
	"interpretation",
] as const;

export const ASSERTION_TYPE = [
	"recurring_issue",
	"recent_breakthrough",
	"student_reported",
	"piece_status",
	"baseline_shift",
] as const;

export type AssertionType = (typeof ASSERTION_TYPE)[number];

export const factSchema = z
	.object({
		id: z.string().uuid(),
		studentId: z.string().min(1),
		factText: z.string().min(1),
		assertionType: z.enum(ASSERTION_TYPE),
		dimension: z.enum(SIX_DIM).nullable().optional(),
		validAt: z.string().datetime(),
		invalidAt: z.string().datetime().nullable(),
		entityMentions: z.array(entityRefSchema).min(1),
		evidence: z.array(evidenceRefSchema).min(1),
		trend: z
			.enum(["improving", "stable", "declining", "new"])
			.nullable()
			.optional(),
		confidence: z.enum(["high", "medium", "low"]),
		sourceType: z.enum(["synthesized", "student_reported", "inferred"]),
		createdAt: z.string().datetime(),
		expiredAt: z.string().datetime().nullable(),
	})
	.refine(
		(f) =>
			f.invalidAt === null ||
			Date.parse(f.invalidAt) >= Date.parse(f.validAt),
		{ message: "invalidAt must be >= validAt", path: ["invalidAt"] },
	)
	.refine(
		(f) =>
			f.expiredAt === null ||
			Date.parse(f.expiredAt) >= Date.parse(f.createdAt),
		{ message: "expiredAt must be >= createdAt", path: ["expiredAt"] },
	);

export type Fact = z.infer<typeof factSchema>;
