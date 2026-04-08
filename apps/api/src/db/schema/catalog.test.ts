import { describe, expect, it } from "vitest";
import { pieces } from "./catalog";

describe("pieces schema", () => {
	it("has opusNumber, pieceNumber, catalogueType columns defined", () => {
		// TypeScript compile check: if these columns don't exist, this file won't compile.
		// Runtime check: Drizzle column objects are defined (not undefined).
		expect(pieces.opusNumber).toBeDefined();
		expect(pieces.pieceNumber).toBeDefined();
		expect(pieces.catalogueType).toBeDefined();
	});
});
