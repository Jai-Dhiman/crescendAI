import { render, screen } from "@testing-library/react";
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

	it("positions a mark at its share of the session duration", () => {
		render(
			<SessionTimelineStrip
				durationSeconds={FIXTURE_DURATION_SECONDS}
				marks={FIXTURE_MARKS}
			/>,
		);

		// m1 is at 64s of 360s = 17.777...%
		const wrapper = screen.getByLabelText(/Pedaling/).parentElement;
		expect(wrapper).toHaveStyle({ left: `${(64 / 360) * 100}%` });
	});
});
