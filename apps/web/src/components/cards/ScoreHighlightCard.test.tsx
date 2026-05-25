// src/components/cards/ScoreHighlightCard.test.tsx
import { render, screen, waitFor } from "@testing-library/react";
import * as React from "react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import type { ScoreHighlightConfig } from "../../lib/types";

const mockGetClip = vi.fn();
vi.mock("../../lib/score-renderer", () => ({
	scoreRenderer: {
		getClip: (...args: unknown[]) => mockGetClip(...args),
	},
}));

beforeEach(() => {
	vi.clearAllMocks();
});

describe("ScoreHighlightCard", () => {
	it("renders dimension label, bar range, and annotation when getClip rejects", async () => {
		mockGetClip.mockRejectedValue(new Error("Worker unavailable"));
		const config: ScoreHighlightConfig = {
			pieceId: "chopin.ballades.1",
			highlights: [
				{
					bars: [1, 4] as [number, number],
					dimension: "dynamics",
					annotation: "hushed opening",
				},
			],
		};
		const { ScoreHighlightCard } = await import("./ScoreHighlightCard");
		render(React.createElement(ScoreHighlightCard, { config }));
		await waitFor(() => {
			expect(screen.getByText("dynamics")).toBeInTheDocument();
			expect(screen.getByText(/bars 1/)).toBeInTheDocument();
			expect(screen.getByText("hushed opening")).toBeInTheDocument();
			expect(mockGetClip).toHaveBeenCalledWith("chopin.ballades.1", 1, 4);
		});
	});

	it("issues all getClip calls in parallel for multi-highlight configs and renders each SVG", async () => {
		const resolvers: Array<(svg: string) => void> = [];
		mockGetClip.mockImplementation(
			() =>
				new Promise<string>((resolve) => {
					resolvers.push(resolve);
				}),
		);

		const config: ScoreHighlightConfig = {
			pieceId: "chopin.ballades.1",
			highlights: [
				{ bars: [1, 4] as [number, number], dimension: "dynamics", annotation: "a" },
				{ bars: [5, 8] as [number, number], dimension: "timing", annotation: "b" },
				{ bars: [9, 12] as [number, number], dimension: "pedaling", annotation: "c" },
			],
		};
		const { ScoreHighlightCard } = await import("./ScoreHighlightCard");
		render(React.createElement(ScoreHighlightCard, { config }));

		await waitFor(() => {
			expect(mockGetClip).toHaveBeenCalledTimes(3);
		});

		resolvers[0]("<svg data-bars='1-4'></svg>");
		resolvers[1]("<svg data-bars='5-8'></svg>");
		resolvers[2]("<svg data-bars='9-12'></svg>");

		await waitFor(() => {
			const html = document.body.innerHTML;
			expect(html).toContain('data-bars="1-4"');
			expect(html).toContain('data-bars="5-8"');
			expect(html).toContain('data-bars="9-12"');
		});
	});
});
