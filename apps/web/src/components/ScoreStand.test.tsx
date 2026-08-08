import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { scoreRenderer } from "../lib/score-renderer";
import { ScoreStand } from "./ScoreStand";

vi.mock("../lib/score-renderer", () => ({
	scoreRenderer: {
		load: vi.fn(),
		getPage: vi.fn(),
	},
}));

const TWO_PAGE_IR = {
	pieceId: "chopin-nocturne-op9-no2",
	verovioVersion: "test",
	pageWidth: 1000,
	pages: [
		{
			pageN: 1,
			viewBox: "0 0 100 100",
			width: 100,
			height: 100,
			systemBboxes: [],
		},
		{
			pageN: 2,
			viewBox: "0 0 100 100",
			width: 100,
			height: 100,
			systemBboxes: [],
		},
	],
	bars: [
		{
			barNumber: 1,
			measureOn: "m1",
			pageN: 1,
			bbox: { x: 0, y: 0, w: 0, h: 0 },
			noteIds: [],
			qstampStart: 0,
			qstampEnd: 4,
		},
		{
			barNumber: 9,
			measureOn: "m9",
			pageN: 2,
			bbox: { x: 0, y: 0, w: 0, h: 0 },
			noteIds: [],
			qstampStart: 32,
			qstampEnd: 36,
		},
	],
	notes: {},
};

describe("ScoreStand", () => {
	beforeEach(() => {
		vi.mocked(scoreRenderer.load).mockResolvedValue({
			ir: TWO_PAGE_IR,
			pageSvgs: ["<svg data-page='1'></svg>", "<svg data-page='2'></svg>"],
		});
		vi.mocked(scoreRenderer.getPage).mockImplementation(
			async (_pieceId, pageN) => `<svg data-page="${pageN}"></svg>`,
		);
	});

	it("loads page 1 first, then advances and retreats, clamped at both ends", async () => {
		render(
			<ScoreStand
				pieceId="chopin-nocturne-op9-no2"
				marks={[]}
				elapsedSeconds={0}
				isRecording={true}
			/>,
		);

		await waitFor(() =>
			expect(screen.getByTestId("score-stand-page")).toHaveAttribute(
				"data-current-page",
				"1",
			),
		);

		const prevButton = screen.getByRole("button", { name: /previous page/i });
		expect(prevButton).toBeDisabled();

		fireEvent.click(screen.getByRole("button", { name: /next page/i }));
		await waitFor(() =>
			expect(screen.getByTestId("score-stand-page")).toHaveAttribute(
				"data-current-page",
				"2",
			),
		);

		const nextButton = screen.getByRole("button", { name: /next page/i });
		expect(nextButton).toBeDisabled();

		fireEvent.click(screen.getByRole("button", { name: /previous page/i }));
		await waitFor(() =>
			expect(screen.getByTestId("score-stand-page")).toHaveAttribute(
				"data-current-page",
				"1",
			),
		);
	});
});
