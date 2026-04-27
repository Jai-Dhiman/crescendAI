import { z } from "zod";

const SIX_DIM = [
	"dynamics",
	"timing",
	"pedaling",
	"articulation",
	"phrasing",
	"interpretation",
] as const;

const FRAMING = [
	"correction",
	"recognition",
	"encouragement",
	"question",
] as const;

export const ObservationSchema = z.object({
	id: z.string().uuid(),
	studentId: z.string().min(1),
	sessionId: z.string().uuid(),
	chunkIndex: z.number().int().nullable().optional(),
	dimension: z.enum(SIX_DIM),
	observationText: z.string().min(1),
	elaborationText: z.string().nullable().optional(),
	reasoningTrace: z.string().nullable().optional(),
	framing: z.enum(FRAMING).nullable().optional(),
	dimensionScore: z.number().nullable().optional(),
	studentBaseline: z.number().nullable().optional(),
	pieceContext: z.string().nullable().optional(),
	learningArc: z.string().nullable().optional(),
	isFallback: z.boolean(),
	createdAt: z.string().datetime(),
	messageId: z.string().nullable().optional(),
	conversationId: z.string().nullable().optional(),
});

export type Observation = z.infer<typeof ObservationSchema>;
