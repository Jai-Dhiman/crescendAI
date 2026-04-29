import { describe, expect, it } from "vitest";
import { parseTitleFields } from "./catalog-parse";

describe("parseTitleFields", () => {
	it("extracts opus and piece number from standard Op./No. title", () => {
		const result = parseTitleFields("Etudes Op. 10 No. 3");
		expect(result).toEqual({
			opusNumber: 10,
			pieceNumber: 3,
			catalogueType: "op",
		});
	});

	it("extracts opus and piece number from waltz title", () => {
		const result = parseTitleFields("Waltz Op. 64 No. 2");
		expect(result).toEqual({
			opusNumber: 64,
			pieceNumber: 2,
			catalogueType: "op",
		});
	});

	it("extracts piece number without opus for bare No. format", () => {
		const result = parseTitleFields("Ballades No. 1");
		expect(result).toEqual({
			opusNumber: null,
			pieceNumber: 1,
			catalogueType: null,
		});
	});

	it("identifies WTC catalogue type and extracts trailing piece number", () => {
		const result = parseTitleFields("WTC I - Prelude - 1");
		expect(result).toEqual({
			opusNumber: null,
			pieceNumber: 1,
			catalogueType: "wtc",
		});
	});

	it("returns all null for title with no structured identifiers", () => {
		const result = parseTitleFields("Arabesques");
		expect(result).toEqual({
			opusNumber: null,
			pieceNumber: null,
			catalogueType: null,
		});
	});

	it("handles title with opus only and no piece number", () => {
		const result = parseTitleFields("Sonata Op. 27");
		expect(result).toEqual({
			opusNumber: 27,
			pieceNumber: null,
			catalogueType: "op",
		});
	});

	it("handles WTC title without trailing number", () => {
		const result = parseTitleFields("WTC I - Prelude");
		expect(result).toEqual({
			opusNumber: null,
			pieceNumber: null,
			catalogueType: "wtc",
		});
	});
});
