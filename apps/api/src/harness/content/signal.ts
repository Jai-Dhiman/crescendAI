import { z } from "zod";

const SIX_DIM = [
	"dynamics",
	"timing",
	"pedaling",
	"articulation",
	"phrasing",
	"interpretation",
] as const;

export const SignalSchemaName = {
	MuQQuality: "MuQQuality",
	AMTTranscription: "AMTTranscription",
	StopMoment: "StopMoment",
	ScoreAlignment: "ScoreAlignment",
} as const;

export type SignalSchemaName =
	(typeof SignalSchemaName)[keyof typeof SignalSchemaName];

const headers = z.object({
	chunk_id: z.string().min(1),
	producer: z.string().min(1),
	producer_version: z.string().min(1),
	created_at: z.string().datetime(),
});

const muqPayload = z.object({
	dynamics: z.number(),
	timing: z.number(),
	pedaling: z.number(),
	articulation: z.number(),
	phrasing: z.number(),
	interpretation: z.number(),
});

const amtPayload = z.object({
	midi_notes: z.array(
		z.object({
			pitch: z.number().int(),
			onset_ms: z.number(),
			offset_ms: z.number(),
			velocity: z.number().int(),
		}),
	),
	pedals: z.array(
		z.object({
			onset_ms: z.number(),
			offset_ms: z.number(),
			type: z.string(),
		}),
	),
});

const stopPayload = z.object({
	probability: z.number().min(0).max(1),
	dimension: z.enum(SIX_DIM),
	bar_range: z
		.object({ start: z.number().int(), end: z.number().int() })
		.optional(),
});

const scoreAlignPayload = z.object({
	alignments: z.array(
		z.object({
			chunk_offset_ms: z.number(),
			score_offset_ms: z.number(),
			confidence: z.number(),
		}),
	),
});

const muq = headers.extend({
	schema_name: z.literal("MuQQuality"),
	payload: muqPayload,
});

const amt = headers.extend({
	schema_name: z.literal("AMTTranscription"),
	payload: amtPayload,
});

const stop = headers.extend({
	schema_name: z.literal("StopMoment"),
	payload: stopPayload,
});

const align = headers.extend({
	schema_name: z.literal("ScoreAlignment"),
	payload: scoreAlignPayload,
});

export const SignalSchema = z.discriminatedUnion("schema_name", [
	muq,
	amt,
	stop,
	align,
]);

export type Signal = z.infer<typeof SignalSchema>;

export const signalSchemas = {
	MuQQuality: muq,
	AMTTranscription: amt,
	StopMoment: stop,
	ScoreAlignment: align,
} as const;
