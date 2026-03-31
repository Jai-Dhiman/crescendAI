import type { Dimension } from "../lib/dims";
import { DIMS_6 } from "../lib/dims";

export interface AccumulatedMoment {
  chunkIndex: number;
  dimension: Dimension;
  score: number;
  baseline: number;
  deviation: number;
  isPositive: boolean;
  reasoning: string;
  barRange: [number, number] | null;
  analysisTier: number; // 1=full, 2=absolute, 3=scores only
  timestampMs: number;
  llmAnalysis: string | null;
}

export interface ModeTransitionRecord {
  from: string;
  to: string;
  chunkIndex: number;
  timestampMs: number;
  dwellMs: number;
}

export interface DrillingRecord {
  barRange: [number, number] | null;
  repetitionCount: number;
  firstScores: number[];
  finalScores: number[];
  startedAtChunk: number;
  endedAtChunk: number;
}

export interface TimelineEvent {
  chunkIndex: number;
  timestampMs: number;
  hasAudio: boolean;
}

export class SessionAccumulator {
  teachingMoments: AccumulatedMoment[] = [];
  modeTransitions: ModeTransitionRecord[] = [];
  drillingRecords: DrillingRecord[] = [];
  timeline: TimelineEvent[] = [];

  accumulateMoment(moment: AccumulatedMoment): void {
    this.teachingMoments.push(moment);
  }

  accumulateModeTransition(record: ModeTransitionRecord): void {
    this.modeTransitions.push(record);
  }

  accumulateDrillingRecord(record: DrillingRecord): void {
    this.drillingRecords.push(record);
  }

  accumulateTimelineEvent(event: TimelineEvent): void {
    this.timeline.push(event);
  }

  hasTeachingContent(): boolean {
    return this.teachingMoments.length > 0 || this.drillingRecords.length > 0;
  }

  /**
   * Select top teaching moments for synthesis.
   * Per dimension: pick highest |deviation|. Also pick top positive if different.
   * Cap at 8, sort by chunk index (chronological).
   */
  topMoments(dimensionWeights?: Record<string, number>): AccumulatedMoment[] {
    const selected: AccumulatedMoment[] = [];

    for (const dim of DIMS_6) {
      const dimMoments = this.teachingMoments.filter((m) => m.dimension === dim);

      if (dimMoments.length === 0) {
        continue;
      }

      // Top-1 per dimension by |deviation|
      const topByDeviation = dimMoments.reduce((best, m) =>
        Math.abs(m.deviation) > Math.abs(best.deviation) ? m : best,
      );

      if (!selected.includes(topByDeviation)) {
        selected.push(topByDeviation);
      }

      // Top-1 positive per dimension if different from topByDeviation
      const positiveMoments = dimMoments.filter((m) => m.isPositive);
      if (positiveMoments.length > 0) {
        const topPositive = positiveMoments.reduce((best, m) =>
          Math.abs(m.deviation) > Math.abs(best.deviation) ? m : best,
        );

        if (topPositive !== topByDeviation && !selected.includes(topPositive)) {
          selected.push(topPositive);
        }
      }
    }

    // If dimension weights provided, re-sort by weighted |deviation| before capping
    if (dimensionWeights !== undefined) {
      selected.sort((a, b) => {
        const aw = Math.abs(a.deviation) * (dimensionWeights[a.dimension] ?? 1.0);
        const bw = Math.abs(b.deviation) * (dimensionWeights[b.dimension] ?? 1.0);
        return bw - aw;
      });
    }

    // Cap at 8
    const capped = selected.slice(0, 8);

    // Sort by chunkIndex ascending (chronological)
    capped.sort((a, b) => a.chunkIndex - b.chunkIndex);

    return capped;
  }

  toJSON(): unknown {
    return {
      teachingMoments: this.teachingMoments,
      modeTransitions: this.modeTransitions,
      drillingRecords: this.drillingRecords,
      timeline: this.timeline,
    };
  }

  static fromJSON(data: unknown): SessionAccumulator {
    const acc = new SessionAccumulator();

    if (data === null || typeof data !== "object") {
      return acc;
    }

    const d = data as Record<string, unknown>;

    if (Array.isArray(d["teachingMoments"])) {
      acc.teachingMoments = d["teachingMoments"] as AccumulatedMoment[];
    }
    if (Array.isArray(d["modeTransitions"])) {
      acc.modeTransitions = d["modeTransitions"] as ModeTransitionRecord[];
    }
    if (Array.isArray(d["drillingRecords"])) {
      acc.drillingRecords = d["drillingRecords"] as DrillingRecord[];
    }
    if (Array.isArray(d["timeline"])) {
      acc.timeline = d["timeline"] as TimelineEvent[];
    }

    return acc;
  }
}
