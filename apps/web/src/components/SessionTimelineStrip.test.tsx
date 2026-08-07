import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import {
	FIXTURE_DURATION_SECONDS,
	FIXTURE_MARKS,
} from "../test-utils/mark-fixtures";
import { SessionTimelineStrip } from "./SessionTimelineStrip";

describe("SessionTimelineStrip", () => {
	it("renders every mark, including ones the score canvas cannot place", () => {
		const { container } = render(
			<SessionTimelineStrip
				durationSeconds={FIXTURE_DURATION_SECONDS}
				marks={FIXTURE_MARKS}
			/>,
		);

		expect(container.querySelectorAll("button[aria-expanded]")).toHaveLength(
			FIXTURE_MARKS.length,
		);
		// m5 is bar 88 — absent from the score canvas, present here.
		expect(screen.getByLabelText(/Articulation/)).toBeInTheDocument();
		// m4's bars were discarded; it must read as a timestamp.
		expect(
			screen.getByLabelText(/Needs work: Timing, 1:37/),
		).toBeInTheDocument();
	});

	// "a mark sits at its share of the session duration" USED to be asserted
	// here, against a `left: 17.77%` string. It has moved to tests/marks.spec.ts.
	// Position is now derived from the measured strip width so a mark can be
	// held inside its container, and jsdom reports every width as 0 — so this
	// file can no longer produce the fact, only a constant. Asserting a style
	// string that layout never consumed is what let the overflow bug survive.

	it("expands and collapses a mark's evidence on tap", () => {
		render(
			<SessionTimelineStrip
				durationSeconds={FIXTURE_DURATION_SECONDS}
				marks={FIXTURE_MARKS}
			/>,
		);
		const glyph = screen.getByLabelText(/Needs work: Timing, 1:37/);

		expect(screen.queryByText(/the left hand lagged/)).not.toBeInTheDocument();

		fireEvent.click(glyph);
		expect(screen.getByText(/the left hand lagged/)).toBeInTheDocument();
		expect(glyph).toHaveAttribute("aria-expanded", "true");

		fireEvent.click(glyph);
		expect(screen.queryByText(/the left hand lagged/)).not.toBeInTheDocument();
	});
});
