import { test, expect } from "vitest";
import { SegmentLoopArtifactSchema } from "./segment-loop";

const VALID = {
  kind: "segment_loop" as const,
  id: "loop-abc",
  studentId: "stu-1",
  pieceId: "chopin.ballades.1",
  barsStart: 12,
  barsEnd: 16,
  requiredCorrect: 3,
  attemptsCompleted: 0,
  status: "pending" as const,
  dimension: null,
};

test("valid SegmentLoopArtifact passes", () => {
  const r = SegmentLoopArtifactSchema.safeParse(VALID);
  expect(r.success).toBe(true);
});

test("bars_end < bars_start fails refinement", () => {
  const r = SegmentLoopArtifactSchema.safeParse({ ...VALID, barsStart: 20, barsEnd: 10 });
  expect(r.success).toBe(false);
});

test("invalid status rejected", () => {
  const r = SegmentLoopArtifactSchema.safeParse({ ...VALID, status: "unknown" });
  expect(r.success).toBe(false);
});
