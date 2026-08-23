import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { VerdictBanner } from "./VerdictBanner";

describe("VerdictBanner", () => {
	it("renders loading copy, the verdict sentence when ready, and an explicit retry state when failed", () => {
		const { rerender } = render(
			<VerdictBanner verdict={{ status: "loading" }} onRetry={() => {}} />,
		);
		expect(
			screen.getByText(/Putting together your review/),
		).toBeInTheDocument();

		rerender(
			<VerdictBanner
				verdict={{ status: "ready", text: "the bass-blur is gone" }}
				onRetry={() => {}}
			/>,
		);
		expect(screen.getByText("the bass-blur is gone")).toBeInTheDocument();

		const onRetry = vi.fn();
		rerender(
			<VerdictBanner verdict={{ status: "failed" }} onRetry={onRetry} />,
		);
		expect(screen.getByRole("alert")).toBeInTheDocument();
		fireEvent.click(screen.getByRole("button", { name: /try again/i }));
		expect(onRetry).toHaveBeenCalledTimes(1);
	});
});
