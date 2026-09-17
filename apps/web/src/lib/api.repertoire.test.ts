import { afterEach, describe, expect, it, vi } from "vitest";
import { api } from "./api";

afterEach(() => {
	vi.unstubAllGlobals();
});

function stubFetch(body: unknown) {
	const spy = vi.fn().mockResolvedValue(
		new Response(JSON.stringify(body), {
			status: 200,
			headers: { "Content-Type": "application/json" },
		}),
	);
	vi.stubGlobal("fetch", spy);
	return spy;
}

describe("api.repertoire", () => {
	it("lists the student's repertoire", async () => {
		const entry = {
			pieceId: "p1",
			composer: "Chopin",
			title: "Nocturne Op. 9 No. 2",
			sessionCount: 3,
			lastPracticedAt: "2026-08-25T18:00:00.000Z",
			openMarkCount: null,
		};
		const spy = stubFetch({ entries: [entry] });

		const entries = await api.repertoire.list();

		expect(entries).toEqual([entry]);
		expect(spy.mock.calls[0][0]).toMatch(/\/api\/repertoire$/);
	});

	it("fetches one piece's history", async () => {
		const spy = stubFetch({
			pieceId: "p1",
			composer: "Chopin",
			title: "Nocturne",
			sessions: [],
			openDrills: [],
		});

		const history = await api.repertoire.history("p1");

		expect(history.pieceId).toBe("p1");
		expect(spy.mock.calls[0][0]).toMatch(/\/api\/repertoire\/p1$/);
	});

	it("lists the score catalog", async () => {
		const spy = stubFetch({
			pieces: [
				{ pieceId: "p1", composer: "Chopin", title: "Nocturne Op. 9 No. 2" },
			],
		});

		const pieces = await api.scores.list();

		expect(pieces).toHaveLength(1);
		expect(spy.mock.calls[0][0]).toMatch(/\/api\/scores$/);
	});
});
