import { render, waitFor } from "@testing-library/react";
import * as React from "react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import type { ExerciseSetConfig } from "../../lib/types";

const mockGetClip = vi.fn();
vi.mock("../../lib/score-renderer", () => ({
	scoreRenderer: {
		getClip: (...args: unknown[]) => mockGetClip(...args),
	},
}));
vi.mock("../../lib/api", () => ({
	api: { exercises: { assign: vi.fn() } },
}));

beforeEach(() => {
	vi.clearAllMocks();
});

describe("ExerciseSetCard", () => {
	it("renders the SVG string returned by scoreRenderer.getClip when scoreClip is present", async () => {
		mockGetClip.mockResolvedValue("<svg data-test='exercise-clip'></svg>");
		const config: ExerciseSetConfig = {
			targetSkill: "Voicing the melody",
			sourcePassage: "bars 5-8",
			scoreClip: {
				pieceId: "chopin.ballades.1",
				bars: [5, 8],
			},
			exercises: [
				{
					title: "Slow practice",
					instruction: "Half tempo, both hands.",
					focusDimension: "dynamics",
				},
			],
		};
		const { ExerciseSetCard } = await import("./ExerciseSetCard");
		render(React.createElement(ExerciseSetCard, { config }));

		await waitFor(() => {
			expect(mockGetClip).toHaveBeenCalledWith("chopin.ballades.1", 5, 8);
			expect(document.body.innerHTML).toContain('data-test="exercise-clip"');
		});
	});

	it("renders without a clip section when scoreClip is absent", async () => {
		const config: ExerciseSetConfig = {
			targetSkill: "Voicing the melody",
			sourcePassage: "general",
			exercises: [
				{
					title: "Slow practice",
					instruction: "Half tempo, both hands.",
					focusDimension: "dynamics",
				},
			],
		};
		const { ExerciseSetCard } = await import("./ExerciseSetCard");
		render(React.createElement(ExerciseSetCard, { config }));
		await waitFor(() => {
			expect(document.body.textContent).toContain("Voicing the melody");
		});
		expect(mockGetClip).not.toHaveBeenCalled();
	});
});
