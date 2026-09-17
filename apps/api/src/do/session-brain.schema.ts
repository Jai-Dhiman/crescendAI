import { z } from "zod";
import { DIMS_6 } from "../lib/dims";
import { BaselineStateSchema } from "../services/student-baseline";

export const sessionStateSchema = z.object({
	version: z.number().int(),
	sessionId: z.string(),
	studentId: z.string(),
	conversationId: z.string().nullable(),
	chunksInFlight: z.number().int().default(0),
	sessionEnding: z.boolean().default(false),
	synthesisCompleted: z.boolean().default(false),
	finalized: z.boolean().default(false),
	inferenceFailures: z.number().int().default(0),
	accumulator: z.unknown().default({}),
	baselines: z.record(z.string(), z.number()).nullable().default(null),
	baselinesLoaded: z.boolean().default(false),
	scoredChunks: z
		.array(
			z.object({
				chunkIndex: z.number().int(),
				scores: z.array(z.number()),
			}),
		)
		.default([]),
	pieceLocked: z.boolean().default(false),
	pieceIdentification: z
		.object({
			pieceId: z.string(),
			confidence: z.number(),
			method: z.string(),
		})
		.nullable()
		.default(null),
	modeDetector: z.unknown().default(null),
	passageLoopDetector: z.unknown().default(null),
	identificationNoteBuffer: z
		.array(
			z.object({
				pitch: z.number().int(),
				onset: z.number(),
				offset: z.number(),
				velocity: z.number(),
			}),
		)
		.default([]),
	activeAssignment: z
		.object({
			id: z.string(),
			pieceId: z.string(),
			barsStart: z.number().int(),
			barsEnd: z.number().int(),
			requiredCorrect: z.number().int(),
			attemptsCompleted: z.number().int(),
			dimension: z.string().nullable().default(null),
		})
		.nullable()
		.default(null),
	isEvalSession: z.boolean().default(false),
	markBaseline: BaselineStateSchema.nullable().default(null),
	markBaselineLoaded: z.boolean().default(false),
	sessionSamples: z.record(z.string(), z.array(z.number())).default({}),
	emittedMarks: z.array(z.enum(DIMS_6)).default([]),
	noCommentZones: z
		.array(z.object({ chunkIndex: z.number().int(), reason: z.string() }))
		.default([]),
});

export type SessionState = z.infer<typeof sessionStateSchema>;

/** WebSocket incoming message schemas */
export const wsChunkReadySchema = z.object({
	type: z.literal("chunk_ready"),
	index: z.number().int(),
	r2Key: z.string(),
});

export const wsEndSessionSchema = z.object({
	type: z.literal("end_session"),
});

export const wsSetPieceSchema = z.object({
	type: z.literal("set_piece"),
	query: z.string(),
});

export const wsEvalChunkSchema = z.object({
	type: z.literal("eval_chunk"),
	chunk_index: z.number().int(),
	predictions: z.object({
		dynamics: z.number(),
		timing: z.number(),
		pedaling: z.number(),
		articulation: z.number(),
		phrasing: z.number(),
		interpretation: z.number(),
	}),
	midi_notes: z
		.array(
			z.object({
				pitch: z.number().int(),
				onset: z.number(),
				offset: z.number(),
				velocity: z.number(),
			}),
		)
		.default([]),
	pedal_events: z
		.array(
			z.object({
				time: z.number(),
				value: z.number(),
			}),
		)
		.default([]),
});

export const wsIncomingMessageSchema = z.discriminatedUnion("type", [
	wsChunkReadySchema,
	wsEndSessionSchema,
	wsSetPieceSchema,
	wsEvalChunkSchema,
]);

export type WsChunkReady = z.infer<typeof wsChunkReadySchema>;
export type WsEndSession = z.infer<typeof wsEndSessionSchema>;
export type WsSetPiece = z.infer<typeof wsSetPieceSchema>;
export type WsEvalChunk = z.infer<typeof wsEvalChunkSchema>;
export type WsIncomingMessage = z.infer<typeof wsIncomingMessageSchema>;

/** WebSocket outgoing message schemas */
export const wsChunkBarMapSchema = z.object({
	type: z.literal("chunk_bar_map"),
	chunk_index: z.number().int(),
	bar_min: z.number().int(),
	bar_max: z.number().int(),
	bar_per_frame: z.array(z.number().int()),
});

export const wsMarkSchema = z.object({
	type: z.literal("mark"),
	mark: z.object({
		id: z.string(),
		anchor: z.discriminatedUnion("type", [
			z.object({
				type: z.literal("bars"),
				bars: z.tuple([z.number().int(), z.number().int()]),
				atSeconds: z.number(),
			}),
			z.object({ type: z.literal("timestamp"), atSeconds: z.number() }),
		]),
		taxonomy: z.enum(["needs_work", "missed_opportunity", "strong"]),
		dimension: z.enum(DIMS_6),
		evidence: z.string(),
		lifecycle: z.enum(["active", "improving", "resolved"]),
		confidence: z
			.enum(["exploratory", "provisional", "established"])
			.optional(),
	}),
});

export const wsReviewSchema = z.object({
	type: z.literal("review"),
	verdict: z.string().nullable(),
	carry_forward: z.string().nullable(),
	no_comment_zones: z.array(
		z.object({ atSeconds: z.number(), reason: z.string() }),
	),
});

export const wsOutgoingMessageSchema = z.discriminatedUnion("type", [
	wsChunkBarMapSchema,
	wsMarkSchema,
	wsReviewSchema,
]);

export type WsChunkBarMap = z.infer<typeof wsChunkBarMapSchema>;
export type WsMark = z.infer<typeof wsMarkSchema>;
export type WsReview = z.infer<typeof wsReviewSchema>;
export type WsOutgoingMessage = z.infer<typeof wsOutgoingMessageSchema>;

/** Helper to create initial state */
export function createInitialState(
	sessionId: string,
	studentId: string,
	conversationId: string | null,
): SessionState {
	return {
		version: 0,
		sessionId,
		studentId,
		conversationId,
		chunksInFlight: 0,
		sessionEnding: false,
		synthesisCompleted: false,
		finalized: false,
		inferenceFailures: 0,
		accumulator: {},
		baselines: null,
		baselinesLoaded: false,
		scoredChunks: [],
		pieceLocked: false,
		pieceIdentification: null,
		modeDetector: null,
		passageLoopDetector: null,
		identificationNoteBuffer: [],
		activeAssignment: null,
		isEvalSession: false,
		markBaseline: null,
		markBaselineLoaded: false,
		sessionSamples: {},
		emittedMarks: [],
		noCommentZones: [],
	};
}
