import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { ConfirmPieceChip } from "./ConfirmPieceChip";

const guess = {
	pieceId: "chopin-nocturne-op9-no2",
	composer: "Chopin",
	title: "Nocturne Op. 9 No. 2",
	confidence: 0.92,
};

describe("ConfirmPieceChip", () => {
	it("names the guessed piece and dismisses exactly once", () => {
		const onDismiss = vi.fn();
		render(<ConfirmPieceChip guess={guess} onDismiss={onDismiss} />);

		expect(screen.getByText(/Nocturne Op\. 9 No\. 2/)).toBeInTheDocument();

		fireEvent.click(screen.getByRole("button", { name: /dismiss/i }));
		expect(onDismiss).toHaveBeenCalledTimes(1);
	});
});
