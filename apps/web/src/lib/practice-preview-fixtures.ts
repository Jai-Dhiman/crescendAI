import type { Mark } from "../lib/mark";
import { resolveAnchor } from "../lib/mark";

// Ported from the deleted src/test-utils/mark-fixtures.ts (#157). Kept in a
// leaf module with no component imports (unlike practice-preview.tsx itself,
// which pulls in ScoreStand/PracticeMode and, through them, the api/config
// chain) so that tests/marks.spec.ts can import these constants directly
// under Playwright's plain Node module loader without ever evaluating
// src/lib/config.ts's import.meta.env.PROD access -- Playwright's test
// runner is not Vite, so that access throws outside a component-mounted
// context.
//
// Both constants are exported because tests/marks.spec.ts computes its
// expected pixel fraction from these same numbers instead of hardcoding a
// second, driftable copy.
export const PIECELESS_DURATION_SECONDS = 120;

export const FIXTURE_MARKS: readonly Mark[] = [
	{
		id: "fixture-1",
		anchor: resolveAnchor({ atSeconds: 30, alignmentQuality: 0 }),
		taxonomy: "needs_work",
		dimension: "pedaling",
		evidence: "pedal held through the bass change",
		lifecycle: "active",
	},
	{
		id: "fixture-2",
		anchor: resolveAnchor({ atSeconds: 75, alignmentQuality: 0 }),
		taxonomy: "strong",
		dimension: "phrasing",
		evidence: "the rubato in this phrase was well shaped",
		lifecycle: "improving",
	},
	{
		// 85% of PIECELESS_DURATION_SECONDS (102s of 120s) -- kept near the
		// strip's right edge specifically because a mark anchored there once
		// ran 36px past the strip's edge at every viewport (#157 regression).
		id: "fixture-3",
		anchor: resolveAnchor({ atSeconds: 102, alignmentQuality: 0 }),
		taxonomy: "missed_opportunity",
		dimension: "dynamics",
		evidence: "the closing diminuendo flattened out early",
		lifecycle: "active",
	},
];
