// apps/web/src/components/BarScoreChip.test.tsx
import { render, screen, fireEvent } from "@testing-library/react";
import * as React from "react";
import { describe, expect, it, vi } from "vitest";
import type { BarQualityScores } from "../types/landing";

const SCORES: BarQualityScores = {
  dynamics: 0.52,
  timing: 0.78,
  pedaling: 0.60,
  articulation: 0.71,
  phrasing: 0.54,
  interpretation: 0.63,
};

describe("BarScoreChip", () => {
  it("renders a bar for each of the six dimensions", async () => {
    const { BarScoreChip } = await import("./BarScoreChip");
    render(
      React.createElement(BarScoreChip, {
        scores: SCORES,
        barNumber: 4,
        onClose: vi.fn(),
      }),
    );
    // Six dimension labels must be present
    for (const dim of ["dynamics", "timing", "pedaling", "articulation", "phrasing", "interpretation"]) {
      expect(screen.getByText(new RegExp(dim, "i"))).not.toBeNull();
    }
  });

  it("has aria-modal=true on the dialog element", async () => {
    const { BarScoreChip } = await import("./BarScoreChip");
    render(
      React.createElement(BarScoreChip, {
        scores: SCORES,
        barNumber: 4,
        onClose: vi.fn(),
      }),
    );
    const chip = document.querySelector('[data-testid="bar-score-chip"]');
    expect(chip).not.toBeNull();
    expect(chip!.getAttribute("aria-modal")).toBe("true");
  });

  it("calls onClose when Escape key is pressed", async () => {
    const onClose = vi.fn();
    const { BarScoreChip } = await import("./BarScoreChip");
    render(
      React.createElement(BarScoreChip, {
        scores: SCORES,
        barNumber: 4,
        onClose,
      }),
    );
    // The chip container handles Escape
    const chip = document.querySelector('[data-testid="bar-score-chip"]');
    expect(chip).not.toBeNull();
    fireEvent.keyDown(chip!, { key: "Escape" });
    expect(onClose).toHaveBeenCalledTimes(1);
  });
});
