import { z } from "zod";
import { DIMS_6 } from "../../lib/dims";

export const SEGMENT_LOOP_STATUSES = [
	"pending",
	"active",
	"completed",
	"dismissed",
	"superseded",
] as const;

export const SegmentLoopArtifactSchema = z
	.object({
		kind: z.literal("segment_loop"),
		id: z.string().min(1),
		studentId: z.string().min(1),
		pieceId: z.string().min(1),
		barsStart: z.number().int().positive(),
		barsEnd: z.number().int().positive(),
		requiredCorrect: z.number().int().min(1).max(10),
		attemptsCompleted: z.number().int().min(0),
		status: z.enum(SEGMENT_LOOP_STATUSES),
		dimension: z.enum(DIMS_6 as unknown as [string, ...string[]]).nullable(),
	})
	.refine((v) => v.barsEnd >= v.barsStart, {
		message: "barsEnd must be >= barsStart",
		path: ["barsEnd"],
	});

export type SegmentLoopArtifact = z.infer<typeof SegmentLoopArtifactSchema>;

export interface SegmentLoopRef {
	id: string;
	pieceId: string;
	barsStart: number;
	barsEnd: number;
}
