import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import type { Mark } from "../lib/mark";
import { resolveAnchor } from "../lib/mark";

vi.mock("./MarkPassagePlayback", () => ({
	MarkPassagePlayback: () => <div data-testid="playback-stub" />,
}));

const mark: Mark = {
	id: "m1",
	anchor: resolveAnchor({ atSeconds: 20, bars: [5, 6], alignmentQuality: 1 }),
	taxonomy: "needs_work",
	dimension: "pedaling",
	evidence: "pedal held through the bass change",
	lifecycle: "active",
};

describe("MarkReviewDetail", () => {
	it("shows evidence and the playback region, and fires onWorkOnThis with the mark", async () => {
		const { MarkReviewDetail } = await import("./MarkReviewDetail");
		const onWorkOnThis = vi.fn();
		const onClose = vi.fn();

		render(
			<MarkReviewDetail
				mark={mark}
				sessionId="s1"
				onClose={onClose}
				onWorkOnThis={onWorkOnThis}
			/>,
		);

		expect(
			screen.getByText("pedal held through the bass change"),
		).toBeInTheDocument();
		expect(screen.getByTestId("playback-stub")).toBeInTheDocument();

		fireEvent.click(screen.getByRole("button", { name: /work on this/i }));
		expect(onWorkOnThis).toHaveBeenCalledWith(mark);

		fireEvent.click(screen.getByRole("button", { name: /close/i }));
		expect(onClose).toHaveBeenCalledTimes(1);
	});
});
