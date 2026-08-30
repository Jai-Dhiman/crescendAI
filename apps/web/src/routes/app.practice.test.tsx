import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { AppPractice } from "./app.practice";

vi.mock("@tanstack/react-router", async (importOriginal) => ({
	...(await importOriginal<typeof import("@tanstack/react-router")>()),
	useNavigate: () => vi.fn(),
}));

// PracticeMode's real subtree pulls in Verovio; the ladder decision this test
// checks is visible from PracticeMode's props alone.
vi.mock("../components/PracticeMode", () => ({
	PracticeMode: (props: { userPickedPieceId: string | null }) => (
		<div
			data-testid="practice-mode"
			data-picked-piece={props.userPickedPieceId ?? "none"}
		/>
	),
}));

describe("AppPractice", () => {
	it("puts the picked piece on the stand", () => {
		render(<AppPractice pieceId="chopin-nocturne-op9-no2" />);
		// The start screen precedes the stand; the piece must survive it.
		fireEvent.click(screen.getByRole("button", { name: /start practicing/i }));
		expect(screen.getByTestId("practice-mode")).toHaveAttribute(
			"data-picked-piece",
			"chopin-nocturne-op9-no2",
		);
	});

	it("starts a pieceless session when no piece was picked", () => {
		render(<AppPractice pieceId={null} />);
		fireEvent.click(screen.getByRole("button", { name: /start practicing/i }));
		expect(screen.getByTestId("practice-mode")).toHaveAttribute(
			"data-picked-piece",
			"none",
		);
	});
});
