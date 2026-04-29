// src/components/cards/ScoreHighlightCard.test.tsx
import { render, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import * as React from "react";
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

	it("renders dimension label, bar range, and annotation when getClip rejects", async () => {
		mockGetClip.mockRejectedValue(new Error("Worker unavailable"));
		const { ScoreHighlightCard } = await import("./ScoreHighlightCard");
		render(React.createElement(ScoreHighlightCard, { config }));
		await waitFor(() => {
			expect(screen.getByText("dynamics")).toBeInTheDocument();
			expect(screen.getByText(/bars 1/)).toBeInTheDocument();
			expect(screen.getByText("hushed opening")).toBeInTheDocument();
			expect(mockGetClip).toHaveBeenCalledWith("chopin.ballades.1", 1, 4);
		});
	});
});
