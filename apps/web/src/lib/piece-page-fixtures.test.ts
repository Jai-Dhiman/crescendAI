import { describe, expect, it } from "vitest";
import {
	PIECE_PAGE_ALL_MARKS,
	PIECE_PAGE_MARKS_SPAN_SECONDS,
	PIECE_PAGE_SESSIONS,
} from "./piece-page-fixtures";

describe("piece page fixtures", () => {
	it("carries at least two dated sessions whose marks span all three lifecycle states", () => {
		expect(PIECE_PAGE_SESSIONS.length).toBeGreaterThanOrEqual(2);

		// Distinct sessions, oldest first.
		const ids = PIECE_PAGE_SESSIONS.map((s) => s.sessionId);
		expect(new Set(ids).size).toBe(ids.length);
		const times = PIECE_PAGE_SESSIONS.map((s) => Date.parse(s.startedAt));
		expect(times).toEqual([...times].sort((a, b) => a - b));

		// The accumulated view spans every lifecycle state, which is what the
		// piece page renders as the fade.
		const lifecycles = new Set(
			PIECE_PAGE_SESSIONS.flatMap((s) => s.marks.map((m) => m.lifecycle)),
		);
		expect(lifecycles).toEqual(new Set(["active", "improving", "resolved"]));
	});

	it("tracks one passage improving across the two sessions", () => {
		const [older, newer] = PIECE_PAGE_SESSIONS;
		const olderPedaling = older.marks.find((m) => m.dimension === "pedaling");
		const newerPedaling = newer.marks.find((m) => m.dimension === "pedaling");
		expect(olderPedaling?.lifecycle).toBe("active");
		expect(newerPedaling?.lifecycle).toBe("resolved");
	});

	it("anchors every mark through resolveAnchor, so bar anchors carry atSeconds", () => {
		for (const session of PIECE_PAGE_SESSIONS) {
			for (const mark of session.marks) {
				expect(typeof mark.anchor.atSeconds).toBe("number");
			}
		}
	});

	it("spans far enough to hold every mark, so none is clamped off the strip", () => {
		const latest = Math.max(
			...PIECE_PAGE_ALL_MARKS.map((m) => m.anchor.atSeconds),
		);
		expect(PIECE_PAGE_MARKS_SPAN_SECONDS).toBeGreaterThanOrEqual(latest);
	});
});
