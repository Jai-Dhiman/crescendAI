import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import type { Mark } from "../lib/mark";
import { resolveAnchor } from "../lib/mark";
import { PieceLessMode } from "./PieceLessMode";

const marks: Mark[] = [
	{
		id: "m1",
		anchor: resolveAnchor({ atSeconds: 30, alignmentQuality: 0 }),
		taxonomy: "needs_work",
		dimension: "pedaling",
		evidence: "test evidence",
		lifecycle: "active",
	},
];

describe("PieceLessMode", () => {
	it("shows elapsed time as m:ss and renders the timeline strip with the given marks", () => {
		render(
			<PieceLessMode
				marks={marks}
				durationSeconds={90}
				elapsedSeconds={65}
				isRecording={true}
			/>,
		);

		expect(screen.getByText("1:05")).toBeInTheDocument();
		expect(screen.getByTestId("session-timeline")).toBeInTheDocument();
		expect(
			screen.getByRole("button", { name: /Needs work: Pedaling/i }),
		).toBeInTheDocument();
	});
});
