import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { SessionEndedBanner } from "./SessionEndedBanner";

describe("SessionEndedBanner", () => {
	it("shows the soft-stop state and resumes exactly once", () => {
		const onResume = vi.fn();
		render(<SessionEndedBanner onResume={onResume} />);

		expect(screen.getByText(/Session ended/i)).toBeInTheDocument();

		fireEvent.click(screen.getByRole("button", { name: /keep playing/i }));
		expect(onResume).toHaveBeenCalledTimes(1);
	});
});
