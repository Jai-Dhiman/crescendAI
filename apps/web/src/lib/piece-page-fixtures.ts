import type { Mark } from "./mark";
import { resolveAnchor } from "./mark";

/**
 * Fixture data for the piece page (#160). Marks have no persisted store yet
 * -- they live only in the DO and the session WebSocket -- so "the score with
 * marks accumulated across sessions" is fixture-driven by design, exactly as
 * #160's success criterion words it. The mark-persistence follow-up replaces
 * this module's consumers, not its shape.
 *
 * Every anchor is built by resolveAnchor rather than an object literal: the
 * MarkAnchor brand symbol is unexported precisely so that "a wrong bar number
 * cannot reach the screen" is a compile-time property, and a fixture casting
 * its way past it would be the first hole in that guarantee.
 *
 * Leaf module with no component imports, matching practice-preview-fixtures.ts
 * -- see that file's comment for why (Playwright's loader is not Vite and
 * cannot evaluate lib/config.ts's import.meta.env.PROD access).
 */

export interface FixtureSession {
	readonly sessionId: string;
	/** ISO 8601 */
	readonly startedAt: string;
	readonly marks: readonly Mark[];
}

/** Oldest first. */
export const PIECE_PAGE_SESSIONS: readonly FixtureSession[] = [
	{
		sessionId: "fixture-session-1",
		startedAt: "2026-08-11T18:00:00.000Z",
		marks: [
			{
				id: "pp-1-pedaling",
				anchor: resolveAnchor({
					atSeconds: 64,
					bars: [5, 6],
					alignmentQuality: 1,
				}),
				taxonomy: "needs_work",
				dimension: "pedaling",
				evidence: "pedal held through the bass change at 5.3",
				lifecycle: "active",
			},
			{
				id: "pp-1-dynamics",
				anchor: resolveAnchor({ atSeconds: 120, alignmentQuality: 0 }),
				taxonomy: "missed_opportunity",
				dimension: "dynamics",
				evidence: "the closing diminuendo flattened out early",
				lifecycle: "active",
			},
		],
	},
	{
		sessionId: "fixture-session-2",
		startedAt: "2026-08-18T18:00:00.000Z",
		marks: [
			{
				id: "pp-2-pedaling",
				anchor: resolveAnchor({
					atSeconds: 61,
					bars: [5, 6],
					alignmentQuality: 1,
				}),
				taxonomy: "strong",
				dimension: "pedaling",
				evidence: "the bass change is clean now",
				lifecycle: "resolved",
			},
			{
				id: "pp-2-dynamics",
				anchor: resolveAnchor({ atSeconds: 118, alignmentQuality: 0 }),
				taxonomy: "needs_work",
				dimension: "dynamics",
				evidence: "the diminuendo reaches further, still peaks early",
				lifecycle: "improving",
			},
		],
	},
];

/** Every mark the piece has accumulated, newest session last. */
export const PIECE_PAGE_ALL_MARKS: readonly Mark[] =
	PIECE_PAGE_SESSIONS.flatMap((s) => s.marks);

/**
 * The span the piece page's timeline covers. Derived from the marks rather
 * than hardcoded, so adding a later mark cannot silently push it off the end
 * of the strip -- the clamping bug #157 fixed reappears the moment a mark's
 * atSeconds exceeds the declared duration.
 */
export const PIECE_PAGE_MARKS_SPAN_SECONDS: number = Math.max(
	...PIECE_PAGE_ALL_MARKS.map((m) => m.anchor.atSeconds),
);
