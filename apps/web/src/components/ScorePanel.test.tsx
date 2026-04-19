// src/components/ScorePanel.test.tsx
import { render, waitFor } from "@testing-library/react";
import * as React from "react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { useScorePanelStore } from "../stores/score-panel";

const mockGetFull = vi.fn().mockResolvedValue("<svg><g class='measure'/></svg>");

vi.mock("../lib/score-renderer", () => ({
  scoreRenderer: {
    getFull: (...args: unknown[]) => mockGetFull(...args),
  },
}));

afterEach(() => {
  vi.clearAllMocks();
  useScorePanelStore.getState().clear();
});

describe("ScorePanel", () => {
  it("reads highlightData from store when opened via openHighlight", () => {
    const store = useScorePanelStore.getState();
    store.openHighlight({
      pieceId: "piece-abc",
      highlights: [
        { bars: [4, 8] as [number, number], dimension: "dynamics", annotation: "forte" },
      ],
    });

    const state = useScorePanelStore.getState();
    expect(state.isOpen).toBe(true);
    expect(state.highlightData).not.toBeNull();
    expect(state.highlightData!.pieceId).toBe("piece-abc");
    expect(state.sessionData).toBeNull();
  });

  it("calls scoreRenderer.getFull with pieceId when panel opens with highlight data", async () => {
    useScorePanelStore.getState().openHighlight({
      pieceId: "chopin.ballades.1",
      highlights: [
        { bars: [1, 4] as [number, number], dimension: "dynamics", annotation: "hushed opening" },
      ],
    });

    const { ScorePanel } = await import("./ScorePanel");
    render(React.createElement(ScorePanel));

    await waitFor(() => {
      expect(mockGetFull).toHaveBeenCalledWith("chopin.ballades.1");
    });
  });
});
