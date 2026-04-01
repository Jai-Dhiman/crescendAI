import { z } from "zod";

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
  followerState: z
    .object({
      lastKnownBar: z.number().int().nullable(),
    })
    .default({ lastKnownBar: null }),
  modeDetector: z.unknown().default(null),
  identificationNoteCount: z.number().int().default(0),
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

export const wsIncomingMessageSchema = z.discriminatedUnion("type", [
  wsChunkReadySchema,
  wsEndSessionSchema,
  wsSetPieceSchema,
]);

export type WsChunkReady = z.infer<typeof wsChunkReadySchema>;
export type WsEndSession = z.infer<typeof wsEndSessionSchema>;
export type WsSetPiece = z.infer<typeof wsSetPieceSchema>;
export type WsIncomingMessage = z.infer<typeof wsIncomingMessageSchema>;

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
    followerState: { lastKnownBar: null },
    modeDetector: null,
    identificationNoteCount: 0,
  };
}
