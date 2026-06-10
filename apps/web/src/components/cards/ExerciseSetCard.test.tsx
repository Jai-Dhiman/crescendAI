import { cleanup, render, waitFor } from "@testing-library/react";
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
	cleanup();
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

	it("renders card without clip preview when getClip rejects", async () => {
		mockGetClip.mockRejectedValue(new Error("load failed"));
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
		const { container } = render(React.createElement(ExerciseSetCard, { config }));
		await waitFor(() => {
			expect(mockGetClip).toHaveBeenCalledWith("chopin.ballades.1", 5, 8);
		});
		// Card still renders with the exercise content
		expect(container.textContent).toContain("Voicing the melody");
		// No SVG injected in this render
		expect(container.innerHTML).not.toContain("data-test");
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

	it("renders without crashing when scoreClip and exerciseId are absent (corpus_drill stub)", async () => {
		const config: ExerciseSetConfig = {
			sourcePassage: "bars 1-8",
			targetSkill: "timing focus",
			exercises: [
				{
					title: "Timing corpus drill",
					instruction: "Timing drill coming soon. Practice bars 1-8 at 80% tempo.",
					focusDimension: "timing",
					// no exerciseId — corpus_drill path
				},
			],
			// no scoreClip
		};
		const { ExerciseSetCard } = await import("./ExerciseSetCard");
		expect(() => render(React.createElement(ExerciseSetCard, { config }))).not.toThrow();
		await waitFor(() => {
			expect(document.body.textContent).toContain("timing focus");
			expect(document.body.textContent).toContain("Timing corpus drill");
		});
		expect(mockGetClip).not.toHaveBeenCalled();
	});

	it("renders with scoreClip present (own_passage_loop path) — existing behavior preserved", async () => {
		mockGetClip.mockResolvedValue("<svg data-test='passage-loop-clip'></svg>");
		const config: ExerciseSetConfig = {
			sourcePassage: "bars 12-16",
			targetSkill: "pedaling focus",
			scoreClip: { pieceId: "chopin.ballade.1", bars: [12, 16] },
			exercises: [
				{
					title: "Own passage loop: pedaling",
					instruction: "Loop bars 12-16 at 75% tempo.",
					focusDimension: "pedaling",
					exerciseId: "ex-id-1",
				},
			],
		};
		const { ExerciseSetCard } = await import("./ExerciseSetCard");
		expect(() => render(React.createElement(ExerciseSetCard, { config }))).not.toThrow();
		await waitFor(() => {
			expect(document.body.textContent).toContain("pedaling focus");
		});
	});
});
