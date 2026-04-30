import { render, screen } from "@testing-library/react";
import { test, expect } from "vitest";
import { SegmentLoopArtifactCard } from "./SegmentLoopArtifact";

const CONFIG = {
  id: "loop-1",
  pieceId: "chopin.ballades.1",
  barsStart: 12,
  barsEnd: 16,
  requiredCorrect: 5,
  attemptsCompleted: 2,
  status: "active",
  dimension: null,
};

test("renders bars and attempt counter", () => {
  render(<SegmentLoopArtifactCard config={CONFIG} />);
  expect(screen.getByText(/bars 12.{0,5}16/i)).toBeInTheDocument();
  expect(screen.getByText(/2\s*\/\s*5/)).toBeInTheDocument();
});

test("pending status shows Accept and Skip buttons", () => {
  render(<SegmentLoopArtifactCard config={{ ...CONFIG, status: "pending" }} />);
  expect(screen.getByRole("button", { name: /accept/i })).toBeInTheDocument();
  expect(screen.getByRole("button", { name: /skip/i })).toBeInTheDocument();
});

test("completed status shows completion message", () => {
  render(<SegmentLoopArtifactCard config={{ ...CONFIG, status: "completed" }} />);
  expect(screen.getByText(/complete/i)).toBeInTheDocument();
});
