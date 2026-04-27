import { z } from "zod";

export const PieceSchema = z.object({
	pieceId: z.string().min(1),
	composer: z.string().min(1),
	title: z.string().min(1),
	keySignature: z.string().nullable().optional(),
	timeSignature: z.string().nullable().optional(),
	tempoBpm: z.number().int().nullable().optional(),
	barCount: z.number().int().nonnegative(),
	durationSeconds: z.number().nullable().optional(),
	noteCount: z.number().int().nonnegative(),
	pitchRangeLow: z.number().int().nullable().optional(),
	pitchRangeHigh: z.number().int().nullable().optional(),
	hasTimeSigChanges: z.boolean(),
	hasTempoChanges: z.boolean(),
	source: z.string(),
	opusNumber: z.number().int().nullable().optional(),
	pieceNumber: z.number().int().nullable().optional(),
	catalogueType: z.string().nullable().optional(),
	createdAt: z.string(),
});

export type Piece = z.infer<typeof PieceSchema>;

export const MovementRefSchema = z.object({
	pieceId: z.string().min(1),
	movementIndex: z.number().int().nonnegative(),
});

export type MovementRef = z.infer<typeof MovementRefSchema>;

export const BarRefSchema = z.object({
	pieceId: z.string().min(1),
	movementIndex: z.number().int().nonnegative(),
	barNumber: z.number().int().nonnegative(),
});

export type BarRef = z.infer<typeof BarRefSchema>;

export function pieceIdFromCatalogue(input: {
	composer: string;
	catalogueType: string;
	opusNumber: number;
	pieceNumber: number;
}): string {
	const composer = input.composer.trim().toLowerCase();
	const catalogue = input.catalogueType.trim().toLowerCase();
	return `${composer}.${catalogue}_op_${input.opusNumber}.${input.pieceNumber}`;
}
