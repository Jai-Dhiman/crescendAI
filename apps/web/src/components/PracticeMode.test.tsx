import { act, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { AUTO_STOP_SILENCE_MS } from "../lib/pause-state";
import { PracticeMode } from "./PracticeMode";

vi.mock("../lib/score-renderer", () => ({
	scoreRenderer: {
		load: vi.fn().mockResolvedValue({
			ir: {
				pieceId: "p1",
				verovioVersion: "test",
				pageWidth: 1000,
				pages: [
					{
						pageN: 1,
						viewBox: "0 0 1 1",
						width: 1,
						height: 1,
						systemBboxes: [],
					},
				],
				bars: [],
				notes: {},
			},
			pageSvgs: ["<svg></svg>"],
		}),
		getPage: vi.fn().mockResolvedValue("<svg></svg>"),
	},
}));

const guess = {
	pieceId: "chopin-nocturne-op9-no2",
	composer: "Chopin",
	title: "Nocturne Op. 9 No. 2",
	confidence: 0.92,
};

describe("PracticeMode", () => {
	beforeEach(() => {
		vi.useFakeTimers();
	});
	afterEach(() => {
		vi.useRealTimers();
	});

	it("shows PieceLessMode with no pick and no guess", () => {
		render(
			<PracticeMode
				userPickedPieceId={null}
				confidentGuess={null}
				marks={[]}
				elapsedSeconds={0}
				isPlaying={true}
				isRecording={true}
				onStop={vi.fn()}
			/>,
		);
		expect(screen.getByTestId("session-timeline")).toBeInTheDocument();
		expect(screen.queryByTestId("score-stand-page")).not.toBeInTheDocument();
	});

	it("shows ScoreStand plus a dismissible confirm chip for a confident guess", async () => {
		render(
			<PracticeMode
				userPickedPieceId={null}
				confidentGuess={guess}
				marks={[]}
				elapsedSeconds={0}
				isPlaying={true}
				isRecording={true}
				onStop={vi.fn()}
			/>,
		);
		expect(screen.getByText(/Nocturne Op\. 9 No\. 2/)).toBeInTheDocument();

		fireEvent.click(screen.getByRole("button", { name: /dismiss/i }));
		await vi.waitFor(() =>
			expect(
				screen.queryByText(/Nocturne Op\. 9 No\. 2/),
			).not.toBeInTheDocument(),
		);
	});

	it("shows the session-ended banner after 60s of silence, and resume dismisses it", () => {
		render(
			<PracticeMode
				userPickedPieceId={null}
				confidentGuess={null}
				marks={[]}
				elapsedSeconds={0}
				isPlaying={false}
				isRecording={true}
				onStop={vi.fn()}
			/>,
		);

		act(() => {
			vi.advanceTimersByTime(AUTO_STOP_SILENCE_MS);
		});
		expect(screen.getByText(/Session ended/i)).toBeInTheDocument();

		fireEvent.click(screen.getByRole("button", { name: /keep playing/i }));
		expect(screen.queryByText(/Session ended/i)).not.toBeInTheDocument();
	});

	it("calls onStop exactly once when the stop control is activated, even after auto-stop", () => {
		const onStop = vi.fn();
		render(
			<PracticeMode
				userPickedPieceId={null}
				confidentGuess={null}
				marks={[]}
				elapsedSeconds={0}
				isPlaying={false}
				isRecording={true}
				onStop={onStop}
			/>,
		);

		// Present before auto-stop...
		fireEvent.click(screen.getByRole("button", { name: /stop recording/i }));
		expect(onStop).toHaveBeenCalledTimes(1);

		// ...and still present once the session-ended banner has replaced the
		// rest of the surface -- stopping for real must not require resuming
		// first.
		act(() => {
			vi.advanceTimersByTime(AUTO_STOP_SILENCE_MS);
		});
		expect(screen.getByText(/Session ended/i)).toBeInTheDocument();
		fireEvent.click(screen.getByRole("button", { name: /stop recording/i }));
		expect(onStop).toHaveBeenCalledTimes(2);
	});
});
