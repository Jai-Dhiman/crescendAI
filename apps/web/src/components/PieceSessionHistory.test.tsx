import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import type { PieceSessionSummary } from "../lib/api";
import { PieceSessionHistory } from "./PieceSessionHistory";

const SESSIONS: PieceSessionSummary[] = [
	{
		sessionId: "s2",
		startedAt: "2026-08-25T18:00:00.000Z",
		endedAt: "2026-08-25T18:40:00.000Z",
	},
	{
		sessionId: "s1",
		startedAt: "2026-08-11T18:00:00.000Z",
		endedAt: null,
	},
];

describe("PieceSessionHistory", () => {
	it("lists each session with its date", () => {
		render(<PieceSessionHistory sessions={SESSIONS} />);
		expect(screen.getAllByRole("listitem")).toHaveLength(2);
		expect(screen.getByText("25 Aug 2026")).toBeInTheDocument();
		expect(screen.getByText("11 Aug 2026")).toBeInTheDocument();
	});

	it("shows how long a completed session ran", () => {
		render(<PieceSessionHistory sessions={SESSIONS} />);
		expect(screen.getByText("40 min")).toBeInTheDocument();
	});

	it("says so when the piece has no sessions yet", () => {
		render(<PieceSessionHistory sessions={[]} />);
		expect(screen.getByText("No sessions yet")).toBeInTheDocument();
	});
});
