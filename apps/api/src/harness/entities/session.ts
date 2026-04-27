import { z } from "zod";

export const SessionSchema = z
	.object({
		id: z.string().uuid(),
		studentId: z.string().min(1),
		startedAt: z.string().datetime(),
		endedAt: z.string().datetime().nullable(),
		avgDynamics: z.number().nullable().optional(),
		avgTiming: z.number().nullable().optional(),
		avgPedaling: z.number().nullable().optional(),
		avgArticulation: z.number().nullable().optional(),
		avgPhrasing: z.number().nullable().optional(),
		avgInterpretation: z.number().nullable().optional(),
		observationsJson: z.unknown().nullable().optional(),
		chunksSummaryJson: z.unknown().nullable().optional(),
		conversationId: z.string().nullable().optional(),
		accumulatorJson: z.unknown().nullable().optional(),
		needsSynthesis: z.boolean(),
	})
	.refine(
		(s) => s.endedAt === null || Date.parse(s.endedAt) >= Date.parse(s.startedAt),
		{ message: "endedAt must be >= startedAt" },
	);

export type Session = z.infer<typeof SessionSchema>;
