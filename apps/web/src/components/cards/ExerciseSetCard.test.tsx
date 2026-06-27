import { cleanup, render, waitFor } from "@testing-library/react";
import * as React from "react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import type { ExerciseSetConfig } from "../../lib/types";

const mockGetClip = vi.fn();
const mockGetClipPlayback = vi.fn();
const mockLoad = vi.fn().mockResolvedValue(undefined);

vi.mock("../../lib/score-renderer", () => ({
	scoreRenderer: {
		getClip: (...args: unknown[]) => mockGetClip(...args),
		getClipPlayback: (...args: unknown[]) => mockGetClipPlayback(...args),
		load: (...args: unknown[]) => mockLoad(...args),
	},
}));
vi.mock("../../lib/api", () => ({
	api: { exercises: { assign: vi.fn() } },
}));

vi.mock("../../hooks/useLoopPlayer", () => ({
	useLoopPlayer: vi.fn().mockReturnValue({
		isPlaying: false,
		isCounting: false,
		audioUnavailable: false,
		tempoFactor: 0.75,
		play: vi.fn(),
		pause: vi.fn(),
		stop: vi.fn(),
		setTempoFactor: vi.fn(),
		qstampSource: vi.fn().mockReturnValue(null),
	}),
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
			expect(mockGetClip).toHaveBeenCalledWith("chopin.ballades.1", 5, 8, 0);
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
			expect(mockGetClip).toHaveBeenCalledWith("chopin.ballades.1", 5, 8, 0);
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

	it("renders score as hero and shows transport when scoreClip has tempoFactor", async () => {
		mockGetClipPlayback.mockResolvedValue({
			svg: "<svg data-test='loop-clip'></svg>",
			ir: {
				pieceId: "test",
				verovioVersion: "4.0.0",
				pageWidth: 1600,
				pages: [{ pageN: 1, viewBox: "0 0 1600 600", width: 1600, height: 600, systemBboxes: [] }],
				bars: [
					{ barNumber: 5, measureOn: "m5", pageN: 1, bbox: { x: 0, y: 0, w: 0, h: 0 }, noteIds: [], qstampStart: 16, qstampEnd: 20 },
				],
				notes: {},
			},
			notes: [{ midi: 60, startQ: 16, endQ: 18 }],
		});
		const config: ExerciseSetConfig = {
			sourcePassage: "bars 5-8",
			targetSkill: "dynamics focus",
			scoreClip: { pieceId: "chopin.ballades.1", bars: [5, 8], tempoFactor: 0.75 },
			exercises: [
				{ title: "Loop passage", instruction: "Loop at 75%.", focusDimension: "dynamics" },
			],
		};
		const { ExerciseSetCard } = await import("./ExerciseSetCard");
		render(React.createElement(ExerciseSetCard, { config }));

		await waitFor(() => {
			expect(mockGetClipPlayback).toHaveBeenCalledWith("chopin.ballades.1", 5, 8, 0);
			expect(document.body.innerHTML).toContain('data-test="loop-clip"');
		});
		expect(document.body.querySelector('[data-testid="loop-transport"]')).not.toBeNull();
	});

	it("passes correct wiring props (pieceId, bars, tempoFactor) to useLoopPlayer when scoreClip has tempoFactor", async () => {
		const { useLoopPlayer } = await import("../../hooks/useLoopPlayer");
		const mockUseLoopPlayer = useLoopPlayer as ReturnType<typeof vi.fn>;

		mockGetClipPlayback.mockResolvedValue({
			svg: "<svg data-test='wiring-clip'></svg>",
			ir: {
				pieceId: "bach.wtc.1",
				verovioVersion: "4.0.0",
				pageWidth: 1600,
				pages: [{ pageN: 1, viewBox: "0 0 1600 600", width: 1600, height: 600, systemBboxes: [] }],
				bars: [
					{ barNumber: 1, measureOn: "m1", pageN: 1, bbox: { x: 0, y: 0, w: 0, h: 0 }, noteIds: [], qstampStart: 0, qstampEnd: 4 },
				],
				notes: {},
			},
			notes: [{ midi: 62, startQ: 0, endQ: 2 }],
		});

		const config: ExerciseSetConfig = {
			sourcePassage: "bars 1-4",
			targetSkill: "wiring contract",
			scoreClip: { pieceId: "bach.wtc.1", bars: [1, 4], tempoFactor: 0.8 },
			exercises: [
				{ title: "Wiring test", instruction: "Test.", focusDimension: "timing" },
			],
		};
		const { ExerciseSetCard } = await import("./ExerciseSetCard");
		render(React.createElement(ExerciseSetCard, { config }));

		await waitFor(() => {
			expect(mockGetClipPlayback).toHaveBeenCalledWith("bach.wtc.1", 1, 4, 0);
		});

		// useLoopPlayer must have been called with the correct tempoFactor from scoreClip.
		// This verifies the card-to-hook wiring contract even though the hook itself is mocked.
		expect(mockUseLoopPlayer).toHaveBeenCalled();
		const lastCall = mockUseLoopPlayer.mock.calls[mockUseLoopPlayer.mock.calls.length - 1][0];
		expect(lastCall.tempoFactor).toBe(0.8);
		expect(lastCall.beatsPerBar).toBe(4);
		expect(lastCall.bpmAtUnity).toBe(120);
	});

	it("corpus_drill (no scoreClip, no tempoFactor) renders no transport", async () => {
		const config: ExerciseSetConfig = {
			sourcePassage: "bars 1-8",
			targetSkill: "timing focus",
			exercises: [
				{
					title: "Timing corpus drill",
					instruction: "Timing drill coming soon.",
					focusDimension: "timing",
				},
			],
		};
		const { ExerciseSetCard } = await import("./ExerciseSetCard");
		render(React.createElement(ExerciseSetCard, { config }));
		await waitFor(() => {
			expect(document.body.textContent).toContain("timing focus");
		});
		expect(document.body.querySelector('[data-testid="loop-transport"]')).toBeNull();
		expect(mockGetClipPlayback).not.toHaveBeenCalled();
	});

	it("forwards scoreClip.transpose to load and getClipPlayback", async () => {
		mockGetClipPlayback.mockResolvedValue({
			svg: "<svg data-test='transpose-clip'></svg>",
			ir: {
				pieceId: "hanon_001",
				verovioVersion: "4.0.0",
				pageWidth: 1600,
				pages: [{ pageN: 1, viewBox: "0 0 1600 600", width: 1600, height: 600, systemBboxes: [] }],
				bars: [
					{ barNumber: 1, measureOn: "m1", pageN: 1, bbox: { x: 0, y: 0, w: 0, h: 0 }, noteIds: [], qstampStart: 0, qstampEnd: 4 },
				],
				notes: {},
			},
			notes: [{ midi: 62, startQ: 0, endQ: 2 }],
		});
		const config: ExerciseSetConfig = {
			sourcePassage: "bars 1-29",
			targetSkill: "timing",
			scoreClip: { pieceId: "hanon_001", bars: [1, 29] as [number, number], tempoFactor: 0.8, transpose: 2 },
			exercises: [{ title: "t", instruction: "i", focusDimension: "timing" }],
		};
		const { ExerciseSetCard } = await import("./ExerciseSetCard");
		render(React.createElement(ExerciseSetCard, { config }));
		await waitFor(() => expect(mockGetClipPlayback).toHaveBeenCalled());
		// load(pieceId, transpose) and getClipPlayback(pieceId, b0, b1, transpose).
		expect(mockLoad).toHaveBeenCalledWith("hanon_001", 2);
		expect(mockGetClipPlayback).toHaveBeenCalledWith("hanon_001", 1, 29, 2);
	});
});
