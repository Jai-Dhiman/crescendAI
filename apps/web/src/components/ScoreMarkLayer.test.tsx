import { fireEvent, render, screen } from "@testing-library/react";
import { createRef } from "react";
import { describe, expect, it } from "vitest";
import { FIXTURE_BARS, FIXTURE_MARKS } from "../test-utils/mark-fixtures";
import { ScoreMarkLayer } from "./ScoreMarkLayer";

function renderLayer() {
	const ref = createRef<HTMLDivElement>();
	// Bar 88's element is deliberately absent: it models a bar on a page the
	// overlay is not currently showing.
	const onPage = FIXTURE_BARS.filter((b) => b.barNumber !== 88);
	return render(
		<div ref={ref}>
			{onPage.map((b) => (
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

describe("ScoreMarkLayer", () => {
	it("renders resolvable marks and discloses the count of the rest", () => {
		renderLayer();

		// m1 (bar 5), m2 (bar 12), m3 (bar 30) resolve. m4 and m6 are
		// timestamp-anchored; m5 is bar 88, which is not on this page.
		expect(
			screen.getByLabelText(/Needs work: Pedaling, bars 5-6/),
		).toBeInTheDocument();
		expect(
			screen.getByLabelText(/Missed opportunity: Dynamics, bar 12/),
		).toBeInTheDocument();
		expect(
			screen.getByLabelText(/Strong: Phrasing, bars 30-32/),
		).toBeInTheDocument();

		expect(screen.queryByLabelText(/Articulation/)).not.toBeInTheDocument();
		// Deliberately NOT "not on this page": only m5 is off-page, while m4 and
		// m6 are timestamp-anchored and have no page at all.
		expect(
			screen.getByText("3 marks on the timeline only"),
		).toBeInTheDocument();
	});

	it("expands and collapses a mark's evidence on tap", () => {
		renderLayer();
		const glyph = screen.getByLabelText(/Needs work: Pedaling, bars 5-6/);

		expect(
			screen.queryByText(/pedal held through the bass change/),
		).not.toBeInTheDocument();

		fireEvent.click(glyph);
		expect(
			screen.getByText(/pedal held through the bass change/),
		).toBeInTheDocument();
		expect(glyph).toHaveAttribute("aria-expanded", "true");

		fireEvent.click(glyph);
		expect(
			screen.queryByText(/pedal held through the bass change/),
		).not.toBeInTheDocument();
	});
});
