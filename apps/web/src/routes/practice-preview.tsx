import { createFileRoute } from "@tanstack/react-router";
import { PieceLessMode } from "../components/PieceLessMode";
import { PracticeMode } from "../components/PracticeMode";
import { ScoreStand } from "../components/ScoreStand";
import type { Mark } from "../lib/mark";
import { resolveAnchor } from "../lib/mark";
import type { ConfidentGuess } from "../lib/piece-ladder";
import {
	FIXTURE_MARKS,
	PIECELESS_DURATION_SECONDS,
} from "../lib/practice-preview-fixtures";

export const Route = createFileRoute("/practice-preview")({
	component: PracticePreview,
});

const CONFIRM_CHIP_GUESS: ConfidentGuess = {
	pieceId: "chopin-nocturne-op9-no2",
	composer: "Chopin",
	title: "Nocturne Op. 9 No. 2",
	confidence: 0.92,
};

// This mark MUST be bar-anchored (bars: [1,1]), not timestamp-only:
// mark-placement.ts's placeMarks only ever places marks whose
// anchor.type === "bars"; a "timestamp" anchor never produces a
// data-measure-on glyph. score-ir.ts assigns barNumber: idx + 1, so bar 1
// is guaranteed to exist on page 1 for any score with at least one measure.
const SCORE_FIXTURE_MARKS: readonly Mark[] = [
	{
		id: "score-fixture-1",
		anchor: resolveAnchor({
			atSeconds: 20,
			bars: [1, 1],
			alignmentQuality: 1,
		}),
		taxonomy: "needs_work",
		dimension: "pedaling",
		evidence: "pedal held through the bass change",
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
			<div className="h-1/3 border-b border-border-subtle">
				<ScoreStand
					pieceId="chopin-nocturne-op9-no2"
					marks={SCORE_FIXTURE_MARKS}
					elapsedSeconds={30}
					isRecording={false}
				/>
			</div>
			<div className="h-1/3 border-b border-border-subtle">
				<PieceLessMode
					marks={FIXTURE_MARKS}
					durationSeconds={PIECELESS_DURATION_SECONDS}
					elapsedSeconds={90}
					isRecording={false}
				/>
			</div>
			<div className="h-1/3" data-testid="practice-mode-preview">
				<PracticeMode
					userPickedPieceId={null}
					confidentGuess={CONFIRM_CHIP_GUESS}
					marks={SCORE_FIXTURE_MARKS}
					elapsedSeconds={30}
					isPlaying={true}
					isRecording={true}
					onStop={() => {}}
				/>
			</div>
		</div>
	);
}
