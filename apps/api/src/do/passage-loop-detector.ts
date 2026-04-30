import type { SegmentLoopArtifact } from "../harness/artifacts/segment-loop";

export interface PositionSpan {
  startBar: number;
  endBar: number;
  durationMs: number;
}

export interface LoopAttempt {
  inBounds: boolean;
  ts: number;
  passage: { startBar: number; endBar: number };
}

const TOLERANCE_BARS = 1;
const DEBOUNCE_MS = 2000;

export class PassageLoopDetector {
  private lastEventTs = 0;
  private lastPassageKey = "";

  processPosition(
    span: PositionSpan,
    assignment: SegmentLoopArtifact,
  ): LoopAttempt | null {
    const { startBar, endBar } = span;
    const assignStart = assignment.barsStart;
    const assignEnd = assignment.barsEnd;

    // Strict isolation: start must be within tolerance of assigned start
    const startInWindow =
      startBar >= assignStart - TOLERANCE_BARS &&
      startBar <= assignStart + TOLERANCE_BARS;
    // End must be within tolerance of assigned end
    const endInWindow =
      endBar >= assignEnd - TOLERANCE_BARS && endBar <= assignEnd + TOLERANCE_BARS;

    if (!startInWindow || !endInWindow) return null;

    const passageKey = `${startBar}-${endBar}`;
    const now = Date.now();

    // Debounce: same passage within debounce window counts once
    if (passageKey === this.lastPassageKey && now - this.lastEventTs < DEBOUNCE_MS) {
      return null;
    }

    this.lastEventTs = now;
    this.lastPassageKey = passageKey;

    return { inBounds: true, ts: now, passage: { startBar, endBar } };
  }

  reset(): void {
    this.lastEventTs = 0;
    this.lastPassageKey = "";
  }
}
