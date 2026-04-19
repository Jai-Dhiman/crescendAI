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

vi.mock("../../lib/osmd-manager", () => ({
  osmdManager: {
    ensureRendered: vi.fn().mockResolvedValue(undefined),
    clipBars: vi.fn().mockReturnValue(null),
    reset: vi.fn(),
  },
}));

beforeEach(() => {
  vi.clearAllMocks();
  vi.resetModules();
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
      expect(screen.getAllByText(/1/).length).toBeGreaterThan(0);
      expect(screen.getByText("hushed opening")).toBeInTheDocument();
      expect(mockGetClip).toHaveBeenCalledWith("chopin.ballades.1", 1, 4);
    });
  });
});
