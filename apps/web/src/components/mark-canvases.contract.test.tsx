import { render, screen } from "@testing-library/react";
import { createRef } from "react";
import { describe, expect, it } from "vitest";
import {
	FIXTURE_BARS,
	FIXTURE_DURATION_SECONDS,
	FIXTURE_MARKS,
} from "../test-utils/mark-fixtures";
import { ScoreMarkLayer } from "./ScoreMarkLayer";
import { SessionTimelineStrip } from "./SessionTimelineStrip";

/**
 * Builds a container holding one element per fixture bar that is on the
 * rendered page, with the id ScoreMarkLayer resolves through
 * BarIR.measureOn. Bar 88 is deliberately omitted: it models a bar on a
 * page the overlay is not showing.
 */
function renderScoreCanvas() {
	const ref = createRef<HTMLDivElement>();
	const ON_PAGE = FIXTURE_BARS.filter((b) => b.barNumber !== 88);
	return render(
		<div ref={ref}>
			{ON_PAGE.map((b) => (
				<div key={b.measureOn} id={b.measureOn} />
			))}
			<ScoreMarkLayer
				containerRef={ref}
				bars={FIXTURE_BARS}
				marks={FIXTURE_MARKS}
			/>
		</div>,
	);
}

function namesOf(container: HTMLElement): string[] {
	return Array.from(container.querySelectorAll("button[aria-expanded]")).map(
		(b) => b.getAttribute("aria-label") ?? "",
	);
}

describe("mark canvases share one vocabulary", () => {
	it("renders every mark on the timeline canvas", () => {
		const { container } = render(
			<SessionTimelineStrip
				durationSeconds={FIXTURE_DURATION_SECONDS}
				marks={FIXTURE_MARKS}
			/>,
		);
		expect(namesOf(container)).toHaveLength(FIXTURE_MARKS.length);
	});

	it("gives a mark the identical accessible name on whichever canvas shows it", () => {
		const timeline = render(
			<SessionTimelineStrip
				durationSeconds={FIXTURE_DURATION_SECONDS}
				marks={FIXTURE_MARKS}
			/>,
		);
		const timelineNames = new Set(namesOf(timeline.container));
		timeline.unmount();

		const score = renderScoreCanvas();
		const scoreNames = namesOf(score.container);

		// Canvas A is lossy BY DESIGN — it shows only what it can truly place.
		// So this is containment, not equality. Equality would assert the
		// opposite of the intended behaviour.
		expect(scoreNames.length).toBeGreaterThan(0);
		expect(scoreNames.length).toBeLessThan(FIXTURE_MARKS.length);
		for (const name of scoreNames) {
			expect(timelineNames).toContain(name);
		}
	});

	it("never shows a bar number for a low-alignment mark on either canvas", () => {
		const timeline = render(
			<SessionTimelineStrip
				durationSeconds={FIXTURE_DURATION_SECONDS}
				marks={FIXTURE_MARKS}
			/>,
		);
		// FIXTURE_MARKS m4 supplied bars [21, 22] at alignmentQuality 0.31.
		expect(timeline.container.textContent).not.toContain("21");
		expect(timeline.container.textContent).not.toContain("22");
		expect(screen.getAllByLabelText(/1:37/)).not.toHaveLength(0);
		timeline.unmount();

		const score = renderScoreCanvas();
		expect(score.container.textContent).not.toContain("bars 21");
	});
});
