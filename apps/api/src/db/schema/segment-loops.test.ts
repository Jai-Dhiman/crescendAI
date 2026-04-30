import { test, expect } from "vitest";
import { getTableName } from "drizzle-orm";
import { segmentLoops } from "./segment-loops";

test("segmentLoops table has correct name", () => {
  expect(getTableName(segmentLoops)).toBe("segment_loops");
});

test("segmentLoops table has status, piece_id, student_id columns", () => {
  const cols = Object.keys(segmentLoops);
  expect(cols).toContain("status");
  expect(cols).toContain("pieceId");
  expect(cols).toContain("studentId");
  expect(cols).toContain("barsStart");
  expect(cols).toContain("barsEnd");
  expect(cols).toContain("attemptsCompleted");
  expect(cols).toContain("requiredCorrect");
});
