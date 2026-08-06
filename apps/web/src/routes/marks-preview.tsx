import { createFileRoute } from "@tanstack/react-router";
import { useRef } from "react";
import { ScoreMarkLayer } from "../components/ScoreMarkLayer";
import { SessionTimelineStrip } from "../components/SessionTimelineStrip";
import {
	FIXTURE_BARS,
	FIXTURE_DURATION_SECONDS,
	FIXTURE_MARKS,
} from "../test-utils/mark-fixtures";

export const Route = createFileRoute("/marks-preview")({
	component: MarksPreview,
});

/**
 * Dev preview surface for #157. Deliberately a top-level route rather than a
 * child of /app: /app redirects to /signin when VITE_AUTH_MODE=live, and the
 * a11y run needs to reach this page in a preview build. Removed when the real
 * surfaces (#158/#159/#162) consume the canvases.
 *
 * The measure stand-ins below carry the same ids score-ir emits as
 * BarIR.measureOn, so ScoreMarkLayer's resolution path is exercised for real.
 * Bar 88 is intentionally omitted to exercise the unplaced disclosure.
 */
export function MarksPreview() {
	const scoreRef = useRef<HTMLDivElement>(null);
	const onPage = FIXTURE_BARS.filter((b) => b.barNumber !== 88);

	return (
		<main className="mx-auto max-w-3xl px-6 py-12">
			<h1 className="mb-8 text-display-sm text-ink-primary">
				Mark system preview
			</h1>

			<h2 className="mb-2 text-label-md text-ink-secondary">Score overlay</h2>
			<div
				ref={scoreRef}
				className="score-container relative mb-12 h-64 border border-border-subtle"
			>
				{onPage.map((b, i) => (
					<div
						key={b.measureOn}
						id={b.measureOn}
						className="absolute h-24 w-24 border border-border-subtle"
						style={{ top: 80, left: 24 + i * 140 }}
					/>
				))}
				<ScoreMarkLayer
					containerRef={scoreRef}
					bars={FIXTURE_BARS}
					marks={FIXTURE_MARKS}
				/>
			</div>

			<h2 className="mb-2 text-label-md text-ink-secondary">
				Session timeline
			</h2>
			<SessionTimelineStrip
				durationSeconds={FIXTURE_DURATION_SECONDS}
				marks={FIXTURE_MARKS}
			/>
		</main>
	);
}
