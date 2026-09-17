import {
	cleanup,
	fireEvent,
	render,
	screen,
	waitFor,
} from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import type { Mark } from "../lib/mark";
import { resolveAnchor } from "../lib/mark";
import type { PassageManifest } from "../lib/types";
import { MarkPassagePlayback } from "./MarkPassagePlayback";

const mockGetPassage = vi.fn();
const mockGetClipPlayback = vi.fn();
const mockPlay = vi.fn();
const mockLoad = vi.fn();
const mockDestroy = vi.fn();

vi.mock("../lib/api", () => ({
	api: {
		sessions: {
			getPassage: (...args: unknown[]) => mockGetPassage(...args),
		},
	},
}));

vi.mock("../lib/score-renderer", () => ({
	scoreRenderer: {
		getClipPlayback: (...args: unknown[]) => mockGetClipPlayback(...args),
	},
}));

vi.mock("../lib/passage-player", () => ({
	PassagePlayer: function MockPassagePlayer() {
		return {
			state: "ready",
			duration: 13,
			load: mockLoad,
			play: mockPlay,
			pause: vi.fn(),
			onTick: () => () => undefined,
			destroy: mockDestroy,
		};
	},
}));

vi.mock("../lib/score-cursor", () => ({
	ScoreCursor: function MockScoreCursor() {
		return { start: vi.fn(), stop: vi.fn() };
	},
}));

beforeEach(() => {
	vi.clearAllMocks();
	mockLoad.mockResolvedValue(undefined);
	globalThis.AudioContext = vi.fn(function MockAudioContext() {
		return { close: vi.fn().mockResolvedValue(undefined) };
	}) as unknown as typeof AudioContext;
});

afterEach(() => {
	cleanup();
});

const barMark: Mark = {
	id: "m1",
	anchor: resolveAnchor({ atSeconds: 20, bars: [5, 6], alignmentQuality: 1 }),
	taxonomy: "needs_work",
	dimension: "pedaling",
	evidence: "test evidence",
	lifecycle: "active",
};

const timestampMark: Mark = {
	id: "m2",
	anchor: resolveAnchor({ atSeconds: 20, alignmentQuality: 0 }),
	taxonomy: "needs_work",
	dimension: "pedaling",
	evidence: "test evidence",
	lifecycle: "active",
};

const manifest: PassageManifest = {
	source: { kind: "session", sessionId: "s1" },
	pieceId: "chopin-nocturne-op9-no2",
	bars: [5, 6],
	chunks: [],
	startOffsetSec: 0,
	endOffsetSec: 4,
	barTimeline: [
		{ bar: 5, tSec: 0 },
		{ bar: 6, tSec: 4 },
	],
};

const clipPayload = {
	svg: "<svg data-test='clip'></svg>",
	ir: {
		pieceId: manifest.pieceId,
		verovioVersion: "test",
		pageWidth: 100,
		pages: [],
		bars: [],
		notes: {},
	},
	notes: [],
};

describe("MarkPassagePlayback", () => {
	it("shows no passage for a timestamp-only mark without fetching", async () => {
		render(<MarkPassagePlayback mark={timestampMark} sessionId="s1" />);

		expect(
			screen.getByText(/No passage audio for this mark/),
		).toBeInTheDocument();
		expect(mockGetPassage).not.toHaveBeenCalled();
	});

	it("fetches and renders a Play control for a bar-anchored mark", async () => {
		mockGetPassage.mockResolvedValue(manifest);
		mockGetClipPlayback.mockResolvedValue(clipPayload);

		render(<MarkPassagePlayback mark={barMark} sessionId="s1" />);

		await waitFor(() =>
			expect(
				screen.getByRole("button", { name: /play passage/i }),
			).toBeInTheDocument(),
		);
		expect(mockGetPassage).toHaveBeenCalledWith("s1", [5, 6]);
	});

	it("calls the passage player's play() when the Play button is clicked", async () => {
		mockGetPassage.mockResolvedValue(manifest);
		mockGetClipPlayback.mockResolvedValue(clipPayload);

		render(<MarkPassagePlayback mark={barMark} sessionId="s1" />);

		const playButton = await waitFor(() =>
			screen.getByRole("button", { name: /play passage/i }),
		);
		fireEvent.click(playButton);

		expect(mockPlay).toHaveBeenCalled();
	});

	it("shows an error state when the passage fetch fails", async () => {
		mockGetPassage.mockRejectedValue(new Error("network fail"));

		render(<MarkPassagePlayback mark={barMark} sessionId="s1" />);

		await waitFor(() =>
			expect(screen.getByText(/couldn't load audio/i)).toBeInTheDocument(),
		);
	});

	it("shows audio unavailable when the audio fails to decode, without crashing", async () => {
		mockGetPassage.mockResolvedValue(manifest);
		mockGetClipPlayback.mockResolvedValue(clipPayload);
		mockLoad.mockRejectedValue(new Error("decode fail"));

		render(<MarkPassagePlayback mark={barMark} sessionId="s1" />);

		await waitFor(() =>
			expect(screen.getByText(/Audio unavailable/i)).toBeInTheDocument(),
		);
	});
});
