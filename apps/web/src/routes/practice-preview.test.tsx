import { render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

vi.mock("../lib/score-renderer", () => ({
	scoreRenderer: {
		load: vi.fn().mockResolvedValue("failed"),
		getPage: vi.fn(),
	},
}));

describe("PracticePreview", () => {
	afterEach(() => {
		vi.unstubAllEnvs();
		vi.resetModules();
	});

	it("renders nothing outside dev mode", async () => {
		vi.stubEnv("DEV", false);
		const { PracticePreview } = await import("./practice-preview");
		const { container } = render(<PracticePreview />);
		expect(container).toBeEmptyDOMElement();
	});

	it("renders the pieceless fixture surface in dev mode", async () => {
		vi.stubEnv("DEV", true);
		const { PracticePreview } = await import("./practice-preview");
		render(<PracticePreview />);
		expect(screen.getByTestId("session-timeline")).toBeInTheDocument();
	});

	it("renders the review-ready fixture section with the verdict and carry-forward", async () => {
		vi.stubEnv("DEV", true);
		const { PracticePreview } = await import("./practice-preview");
		render(<PracticePreview />);

		expect(
			screen.getByTestId("practice-mode-review-preview"),
		).toBeInTheDocument();
		expect(
			screen.getByText(/bass-blur from Tuesday is gone/),
		).toBeInTheDocument();
		expect(screen.getByText(/Start with bars 40-44/)).toBeInTheDocument();
	});
});
