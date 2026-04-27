import { describe, expect, test } from "vitest";
import {
	BarRefSchema,
	MovementRefSchema,
	PieceSchema,
	pieceIdFromCatalogue,
} from "./piece";

describe("PieceSchema", () => {
	test("parses a valid Piece row", () => {
		const result = PieceSchema.safeParse({
			pieceId: "chopin.etudes_op_25.1",
			composer: "Chopin",
			title: "Etude Op. 25 No. 1",
			keySignature: "Ab major",
			timeSignature: "4/4",
			tempoBpm: 104,
			barCount: 49,
			durationSeconds: 145,
			noteCount: 1240,
			pitchRangeLow: 36,
			pitchRangeHigh: 96,
			hasTimeSigChanges: false,
			hasTempoChanges: true,
			source: "asap",
			opusNumber: 25,
			pieceNumber: 1,
			catalogueType: "etudes",
			createdAt: "2026-04-01T00:00:00.000Z",
		});
		expect(result.success).toBe(true);
	});

	test("fails parse when barCount is missing", () => {
		const result = PieceSchema.safeParse({
			pieceId: "chopin.etudes_op_25.1",
			composer: "Chopin",
			title: "Etude Op. 25 No. 1",
			noteCount: 1240,
			hasTimeSigChanges: false,
			hasTempoChanges: true,
			source: "asap",
			createdAt: "2026-04-01T00:00:00.000Z",
		});
		expect(result.success).toBe(false);
	});
});

describe("MovementRefSchema", () => {
	test("parses a valid MovementRef", () => {
		const result = MovementRefSchema.safeParse({
			pieceId: "beethoven.sonatas_op_27.2",
			movementIndex: 0,
		});
		expect(result.success).toBe(true);
	});
});

describe("BarRefSchema", () => {
	test("parses a valid BarRef", () => {
		const result = BarRefSchema.safeParse({
			pieceId: "chopin.etudes_op_25.1",
			movementIndex: 0,
			barNumber: 47,
		});
		expect(result.success).toBe(true);
	});
});

describe("pieceIdFromCatalogue", () => {
	test("returns composer.catalogueType_opusNumber.pieceNumber for a known input", () => {
		const id = pieceIdFromCatalogue({
			composer: "Chopin",
			catalogueType: "etudes",
			opusNumber: 25,
			pieceNumber: 1,
		});
		expect(id).toBe("chopin.etudes_op_25.1");
	});
});
