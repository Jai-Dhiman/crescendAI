// apps/web/src/components/cards/PlayPassageCard.test.tsx
import { cleanup, render, screen, waitFor } from "@testing-library/react";
import * as React from "react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import type { PassageManifest, PlayPassageConfig } from "../../lib/types";

const mockGetPassage = vi.fn();
const mockGetClip = vi.fn();
const mockPlay = vi.fn();
const mockLoad = vi.fn();
const mockDestroy = vi.fn();

vi.mock("../../lib/api", () => ({
  api: {
    sessions: {
      getPassage: (...args: unknown[]) => mockGetPassage(...args),
    },
  },
}));

vi.mock("../../lib/score-renderer", () => ({
  scoreRenderer: {
    getClip: (...args: unknown[]) => mockGetClip(...args),
  },
}));

vi.mock("../../lib/passage-player", () => ({
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

describe("PlayPassageCard", () => {
  const config: PlayPassageConfig = {
    sessionId: "00000000-0000-0000-0000-0000000000aa",
    bars: [5, 8],
    focusBars: [6, 7],
    dimension: "timing",
    annotation: "you rushed here",
  };

  const manifest: PassageManifest = {
    source: { kind: "session", sessionId: config.sessionId },
    pieceId: "chopin.ballades.1",
    bars: [5, 8],
    chunks: [{ url: "https://api/c1.webm", chunkIndex: 1, durationSec: 15 }],
    startOffsetSec: 1.0,
    endOffsetSec: 13.0,
    barTimeline: [
      { bar: 5, tSec: 0 },
      { bar: 6, tSec: 4 },
      { bar: 7, tSec: 8 },
      { bar: 8, tSec: 12 },
    ],
  };

  it("renders annotation, dimension label, and bar range after manifest loads", async () => {
    mockGetPassage.mockResolvedValue(manifest);
    mockGetClip.mockResolvedValue({
      svg: "<svg></svg>",
      startMeasureId: null,
      endMeasureId: null,
    });

    const { PlayPassageCard } = await import("./PlayPassageCard");
    render(React.createElement(PlayPassageCard, { config }));

    await waitFor(() => {
      expect(screen.getByText("you rushed here")).toBeInTheDocument();
      expect(screen.getByText("timing")).toBeInTheDocument();
      expect(screen.getByText(/bars 5/)).toBeInTheDocument();
      expect(mockGetPassage).toHaveBeenCalledWith(config.sessionId, [5, 8]);
    });
  });

  it("clicking play invokes PassagePlayer.play()", async () => {
    mockGetPassage.mockResolvedValue(manifest);
    mockGetClip.mockResolvedValue({
      svg: "<svg></svg>",
      startMeasureId: null,
      endMeasureId: null,
    });

    const { PlayPassageCard } = await import("./PlayPassageCard");
    render(React.createElement(PlayPassageCard, { config }));

    const btn = await screen.findByRole("button", { name: /play/i });
    btn.click();
    expect(mockPlay).toHaveBeenCalled();
  });

  it("shows fetch-error state when manifest fetch rejects", async () => {
    mockGetPassage.mockRejectedValue(new Error("getPassage failed: 409"));
    const { PlayPassageCard } = await import("./PlayPassageCard");
    render(React.createElement(PlayPassageCard, { config }));
    await waitFor(() => {
      expect(screen.getByText("couldn't load audio")).toBeInTheDocument();
    });
  });

  it("shows audio_error state — score and annotation render, play button replaced — when player.load() rejects", async () => {
    mockGetPassage.mockResolvedValue(manifest);
    mockGetClip.mockResolvedValue({
      svg: "<svg></svg>",
      startMeasureId: null,
      endMeasureId: null,
    });
    mockLoad.mockRejectedValue(new Error("fetch chunk failed: 404"));

    const { PlayPassageCard } = await import("./PlayPassageCard");
    render(React.createElement(PlayPassageCard, { config }));

    await waitFor(() => {
      expect(screen.getByText("Audio unavailable")).toBeInTheDocument();
      expect(screen.getByText(config.annotation)).toBeInTheDocument();
      expect(screen.queryByRole("button", { name: /play/i })).toBeNull();
      expect(screen.queryByText("couldn't load audio")).toBeNull();
    });
  });
});
