import { createFileRoute } from "@tanstack/react-router";
import { PieceLessMode } from "../components/PieceLessMode";
import type { Mark } from "../lib/mark";
import { resolveAnchor } from "../lib/mark";

export const Route = createFileRoute("/practice-preview")({
	component: PracticePreview,
});

// Ported from the deleted src/test-utils/mark-fixtures.ts (#157). Inlined
// here rather than re-created as a shared test-utils module, because that
// module's whole defect was being importable from a production route in the
// first place -- fixture data now lives only where it is gated out of
// production builds.
//
// Both constants below are exported (not just the component) because
// tests/marks.spec.ts computes its expected pixel fraction from these same
// numbers instead of hardcoding a second, driftable copy.
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

/**
 * Dev-only real-browser harness for #158's successor to #157's
 * marks-preview. Renders null in a production build: import.meta.env.DEV
 * is statically replaced with `false` by Vite in that build, and Rollup's
 * dead-code elimination drops this entire branch -- including the fixture
 * import above -- rather than merely hiding it behind a runtime check.
 *
 * playwright.marks.config.ts's webServer runs `vite dev`, not a production
 * build+preview, specifically so this route is still reachable when
 * tests/marks.spec.ts exercises it.
 */
export function PracticePreview() {
	if (!import.meta.env.DEV) return null;

	return (
		<div className="h-dvh">
			<PieceLessMode
				marks={FIXTURE_MARKS}
				durationSeconds={PIECELESS_DURATION_SECONDS}
				elapsedSeconds={90}
				isRecording={false}
			/>
		</div>
	);
}
