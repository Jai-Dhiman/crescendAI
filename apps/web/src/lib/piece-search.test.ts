import { describe, expect, it } from "vitest";
import type { PieceSummary } from "./piece-search";
import { searchPieces } from "./piece-search";

const CATALOG: readonly PieceSummary[] = [
	{
		pieceId: "chopin-nocturne-op9-no2",
		composer: "Chopin",
		title: "Nocturne Op. 9 No. 2",
	},
	{ pieceId: "chopin-ballade-1", composer: "Chopin", title: "Ballade No. 1" },
	{ pieceId: "bach-invention-1", composer: "Bach", title: "Invention No. 1" },
	{ pieceId: "faure-nocturne-4", composer: "Fauré", title: "Nocturne No. 4" },
];

describe("searchPieces", () => {
	it("returns nothing for an empty or whitespace query", () => {
		expect(searchPieces(CATALOG, "")).toEqual([]);
		expect(searchPieces(CATALOG, "   ")).toEqual([]);
	});

	it("matches on composer regardless of case", () => {
		const found = searchPieces(CATALOG, "chopin");
		expect(found.map((p) => p.pieceId)).toEqual([
			"chopin-nocturne-op9-no2",
			"chopin-ballade-1",
		]);
	});

	it("matches on title", () => {
		const found = searchPieces(CATALOG, "invention");
		expect(found.map((p) => p.pieceId)).toEqual(["bach-invention-1"]);
	});

	it("ranks a prefix match above a mid-string match", () => {
		// "no" prefixes the title "Nocturne..." but only appears mid-string in
		// "Ballade No. 1" and "Invention No. 1".
		const found = searchPieces(CATALOG, "no");
		expect(found[0].title.startsWith("Nocturne")).toBe(true);
	});

	it("matches accented composer names typed without accents", () => {
		const found = searchPieces(CATALOG, "faure");
		expect(found.map((p) => p.pieceId)).toEqual(["faure-nocturne-4"]);
	});

	it("caps results at the requested limit", () => {
		expect(searchPieces(CATALOG, "n", 2)).toHaveLength(2);
	});
});
