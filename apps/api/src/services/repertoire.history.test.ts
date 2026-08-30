import { describe, expect, it } from "vitest";
import { NotFoundError } from "../lib/errors";
import type { Bindings, Db } from "../lib/types";
import { listPieceHistory } from "./repertoire";

const env = {} as Bindings;

/**
 * Drizzle's builder chain has no in-memory backend in this suite, so the
 * chain is stubbed. Assertions are on the returned PieceHistory -- the public
 * result -- never on which builder methods were called.
 */
function makeDb(opts: {
	piece: { pieceId: string; composer: string; title: string } | null;
	sessions: { sessionId: string; startedAt: Date; endedAt: Date | null }[];
	drills: unknown[];
}): Db {
	let call = 0;
	return {
		select: () => ({
			from: () => ({
				where: () => {
					call += 1;
					// 1st: the piece lookup (limit). 2nd: sessions. 3rd: drills.
					if (call === 1) {
						return {
							limit: async () => (opts.piece ? [opts.piece] : []),
						};
					}
					return {
						orderBy: async () => (call === 2 ? opts.sessions : opts.drills),
					};
				},
			}),
		}),
	} as unknown as Db;
}

describe("listPieceHistory", () => {
	it("returns the piece with its sessions newest first and its open drills", async () => {
		const db = makeDb({
			piece: { pieceId: "p1", composer: "Chopin", title: "Nocturne" },
			sessions: [
				{
					sessionId: "s2",
					startedAt: new Date("2026-08-25T18:00:00.000Z"),
					endedAt: new Date("2026-08-25T18:40:00.000Z"),
				},
				{
					sessionId: "s1",
					startedAt: new Date("2026-08-11T18:00:00.000Z"),
					endedAt: null,
				},
			],
			drills: [
				{
					id: "loop-1",
					barsStart: 5,
					barsEnd: 8,
					dimension: "pedaling",
					requiredCorrect: 5,
					attemptsCompleted: 2,
				},
			],
		});

		const history = await listPieceHistory({ db, env }, "student-a", "p1");

		expect(history.pieceId).toBe("p1");
		expect(history.composer).toBe("Chopin");
		expect(history.sessions).toEqual([
			{
				sessionId: "s2",
				startedAt: "2026-08-25T18:00:00.000Z",
				endedAt: "2026-08-25T18:40:00.000Z",
			},
			{
				sessionId: "s1",
				startedAt: "2026-08-11T18:00:00.000Z",
				endedAt: null,
			},
		]);
		expect(history.openDrills).toHaveLength(1);
	});

	it("raises NotFoundError for a piece the catalog does not have", async () => {
		const db = makeDb({ piece: null, sessions: [], drills: [] });
		await expect(
			listPieceHistory({ db, env }, "student-a", "ghost"),
		).rejects.toBeInstanceOf(NotFoundError);
	});
});
