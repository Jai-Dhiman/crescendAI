import { describe, expect, it } from "vitest";
import type { RepertoireRow } from "./repertoire";
import { buildRepertoire } from "./repertoire";

function row(
	pieceId: string,
	composer: string,
	title: string,
	startedAt: string,
): RepertoireRow {
	return { pieceId, composer, title, startedAt: new Date(startedAt) };
}

describe("buildRepertoire", () => {
	it("returns nothing for a student with no sessions", () => {
		expect(buildRepertoire([])).toEqual([]);
	});

	it("collapses repeated sessions of one piece into a single entry", () => {
		const entries = buildRepertoire([
			row("p1", "Chopin", "Nocturne Op. 9 No. 2", "2026-08-11T18:00:00.000Z"),
			row("p1", "Chopin", "Nocturne Op. 9 No. 2", "2026-08-18T18:00:00.000Z"),
			row("p1", "Chopin", "Nocturne Op. 9 No. 2", "2026-08-25T18:00:00.000Z"),
		]);

		expect(entries).toHaveLength(1);
		expect(entries[0]).toEqual({
			pieceId: "p1",
			composer: "Chopin",
			title: "Nocturne Op. 9 No. 2",
			sessionCount: 3,
			lastPracticedAt: "2026-08-25T18:00:00.000Z",
			openMarkCount: null,
		});
	});

	it("takes the most recent session regardless of input order", () => {
		const entries = buildRepertoire([
			row("p1", "Chopin", "Nocturne", "2026-08-25T18:00:00.000Z"),
			row("p1", "Chopin", "Nocturne", "2026-08-11T18:00:00.000Z"),
		]);
		expect(entries[0].lastPracticedAt).toBe("2026-08-25T18:00:00.000Z");
	});

	it("orders pieces by most recently practised first", () => {
		const entries = buildRepertoire([
			row("older", "Bach", "Invention No. 1", "2026-08-01T18:00:00.000Z"),
			row("newer", "Chopin", "Ballade No. 1", "2026-08-26T18:00:00.000Z"),
			row("middle", "Fauré", "Nocturne No. 4", "2026-08-15T18:00:00.000Z"),
		]);
		expect(entries.map((e) => e.pieceId)).toEqual(["newer", "middle", "older"]);
	});

	it("reports openMarkCount as null, never 0", () => {
		// There is no persisted mark store yet. 0 would read to a student as
		// "nothing to work on"; null means "not known", and the card omits it.
		const entries = buildRepertoire([
			row("p1", "Chopin", "Nocturne", "2026-08-25T18:00:00.000Z"),
		]);
		expect(entries[0].openMarkCount).toBeNull();
	});
});
