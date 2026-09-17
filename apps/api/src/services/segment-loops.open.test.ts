import { describe, expect, it } from "vitest";
import type { Db } from "../lib/types";
import { listOpenLoopsForPiece } from "./segment-loops";

/**
 * Captures the where-clause the service builds and returns canned rows. The
 * assertion is on the returned drills -- the public result -- not on the SQL;
 * the stub exists only because Drizzle's builder chain has no in-memory
 * backend in this suite.
 */
function makeDbStub(rows: unknown[]) {
	return {
		select: () => ({
			from: () => ({
				where: () => ({
					orderBy: async () => rows,
				}),
			}),
		}),
	} as unknown as Db;
}

describe("listOpenLoopsForPiece", () => {
	it("maps rows to open drills", async () => {
		const db = makeDbStub([
			{
				id: "loop-1",
				barsStart: 5,
				barsEnd: 8,
				dimension: "pedaling",
				requiredCorrect: 5,
				attemptsCompleted: 2,
			},
			{
				id: "loop-2",
				barsStart: 20,
				barsEnd: 24,
				dimension: null,
				requiredCorrect: 5,
				attemptsCompleted: 0,
			},
		]);

		const drills = await listOpenLoopsForPiece(db, "student-a", "piece-x");

		expect(drills).toEqual([
			{
				id: "loop-1",
				barsStart: 5,
				barsEnd: 8,
				dimension: "pedaling",
				requiredCorrect: 5,
				attemptsCompleted: 2,
			},
			{
				id: "loop-2",
				barsStart: 20,
				barsEnd: 24,
				dimension: null,
				requiredCorrect: 5,
				attemptsCompleted: 0,
			},
		]);
	});

	it("returns an empty list when the piece has no open drills", async () => {
		const drills = await listOpenLoopsForPiece(
			makeDbStub([]),
			"student-a",
			"piece-x",
		);
		expect(drills).toEqual([]);
	});
});
