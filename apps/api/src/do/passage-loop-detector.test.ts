import { describe, expect, test } from "vitest";
import { PassageLoopDetector } from "./passage-loop-detector";
import type { SegmentLoopArtifact } from "../harness/artifacts/segment-loop";

const ASSIGNMENT: SegmentLoopArtifact = {
  kind: "segment_loop",
  id: "loop-1",
  studentId: "stu-1",
  pieceId: "chopin.ballades.1",
  barsStart: 12,
  barsEnd: 16,
  requiredCorrect: 3,
  attemptsCompleted: 0,
  status: "active",
  dimension: null,
};

// PositionSpan: { startBar: number; endBar: number; durationMs: number }
describe("PassageLoopDetector", () => {
  test("clean isolated loop within tolerance returns inBounds=true", () => {
    const det = new PassageLoopDetector();
    const event = det.processPosition(
      { startBar: 12, endBar: 16, durationMs: 8000 },
      ASSIGNMENT,
    );
    expect(event).not.toBeNull();
    expect(event?.inBounds).toBe(true);
  });

  test("start-to-finish playthrough traversing assigned bars returns null", () => {
    const det = new PassageLoopDetector();
    const event = det.processPosition(
      { startBar: 1, endBar: 80, durationMs: 180000 },
      ASSIGNMENT,
    );
    expect(event).toBeNull();
  });

  test("span starting before tolerance window returns null", () => {
    const det = new PassageLoopDetector();
    const event = det.processPosition(
      { startBar: 5, endBar: 16, durationMs: 30000 },
      ASSIGNMENT,
    );
    expect(event).toBeNull();
  });

  test("span ending after tolerance window returns null", () => {
    const det = new PassageLoopDetector();
    const event = det.processPosition(
      { startBar: 12, endBar: 25, durationMs: 30000 },
      ASSIGNMENT,
    );
    expect(event).toBeNull();
  });

  test("same span reported twice is debounced — returns only one event", () => {
    const det = new PassageLoopDetector();
    const span = { startBar: 12, endBar: 16, durationMs: 8000 };
    det.processPosition(span, ASSIGNMENT);
    const second = det.processPosition(span, ASSIGNMENT);
    expect(second).toBeNull();
  });
});
