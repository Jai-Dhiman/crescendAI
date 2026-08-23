import { render, screen } from "@testing-library/react";
import type { ReactNode } from "react";
import { describe, expect, it, vi } from "vitest";
import type { Mark } from "../lib/mark";
import { resolveAnchor } from "../lib/mark";
import { SessionReview } from "./SessionReview";

// vi.mock calls are hoisted above these imports by vitest's transform, so
// SessionReview resolves ScoreStand/PieceLessMode/MarkPassagePlayback to
// the mocks below regardless of write order.
vi.mock("./ScoreStand", () => ({
	ScoreStand: (props: {
		pieceId: string;
		marks: readonly Mark[];
		renderDetail?: (mark: Mark, onClose: () => void) => ReactNode;
	}) => (
		<div data-testid="score-stand" data-piece-id={props.pieceId}>
			{props.renderDetail?.(props.marks[0], () => {})}
		</div>
	),
}));
vi.mock("./PieceLessMode", () => ({
	PieceLessMode: (props: {
		marks: readonly Mark[];
		renderDetail?: (mark: Mark, onClose: () => void) => ReactNode;
	}) => (
		<div data-testid="pieceless-mode">
			{props.renderDetail?.(props.marks[0], () => {})}
		</div>
	),
}));
vi.mock("./MarkPassagePlayback", () => ({
	MarkPassagePlayback: () => <div data-testid="playback-stub" />,
}));

const mark: Mark = {
	id: "m1",
	anchor: resolveAnchor({ atSeconds: 10, bars: [1, 1], alignmentQuality: 1 }),
	taxonomy: "needs_work",
	dimension: "pedaling",
	evidence: "test evidence",
	lifecycle: "active",
};

describe("SessionReview", () => {
	it("renders the score canvas, verdict, and carry-forward when there is a piece", () => {
		render(
			<SessionReview
				pieceId="chopin-nocturne-op9-no2"
				sessionId="s1"
				marks={[mark]}
				durationSeconds={120}
				verdict={{ status: "ready", text: "solid session" }}
				onRetryVerdict={() => {}}
				carryForward={{ text: "start with bars 40-44" }}
				wsGaps={[]}
				noCommentZones={[]}
				onWorkOnThis={() => {}}
			/>,
		);

		expect(screen.getByTestId("score-stand")).toHaveAttribute(
			"data-piece-id",
			"chopin-nocturne-op9-no2",
		);
		expect(screen.getByText("solid session")).toBeInTheDocument();
		expect(screen.getByText(/start with bars 40-44/)).toBeInTheDocument();
		expect(screen.getByText("test evidence")).toBeInTheDocument();
	});

	it("falls back to the pieceless timeline and omits carry-forward when there is none", () => {
		render(
			<SessionReview
				pieceId={null}
				sessionId="s1"
				marks={[mark]}
				durationSeconds={120}
				verdict={{ status: "loading" }}
				onRetryVerdict={() => {}}
				carryForward={null}
				wsGaps={[]}
				noCommentZones={[]}
				onWorkOnThis={() => {}}
			/>,
		);

		expect(screen.getByTestId("pieceless-mode")).toBeInTheDocument();
		expect(screen.queryByText(/Next:/)).not.toBeInTheDocument();
	});
});
