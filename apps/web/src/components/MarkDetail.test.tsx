import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import type { Mark } from "../lib/mark";
import { resolveAnchor } from "../lib/mark";
import { MarkDetail } from "./MarkDetail";

const base: Mark = {
	id: "m1",
	anchor: resolveAnchor({ atSeconds: 97, alignmentQuality: 0 }),
	taxonomy: "needs_work",
	dimension: "timing",
	evidence: "the left hand lagged behind the right through this passage",
	lifecycle: "active",
};

describe("MarkDetail", () => {
	it("shows the evidence and closes on request", () => {
		const onClose = vi.fn();
		render(
			<MarkDetail
				mark={{ ...base, confidence: "established" }}
				onClose={onClose}
			/>,
		);

		expect(
			screen.getByText(/the left hand lagged behind the right/),
		).toBeInTheDocument();
		expect(screen.getByText(/the left hand lagged/)).not.toHaveTextContent(
			"Early read",
		);

		fireEvent.click(screen.getByRole("button", { name: /close/i }));
		expect(onClose).toHaveBeenCalledTimes(1);
	});

	it("frames an exploratory mark as an early read", () => {
		render(
			<MarkDetail
				mark={{ ...base, confidence: "exploratory" }}
				onClose={vi.fn()}
			/>,
		);

		expect(screen.getByText(/Early read/)).toBeInTheDocument();
	});
});
