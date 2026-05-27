// apps/web/src/types/landing.ts

export type BarQualityScores = {
  dynamics: number;
  timing: number;
  pedaling: number;
  articulation: number;
  phrasing: number;
  interpretation: number;
};

export type ProofCardManifest = {
  pieceId: string;
  title: string;
  era: "romantic" | "baroque" | "contemporary";
  audioUrl: string;
  scoreIRUrl: string;
  scoreSvgUrl: string;
  focusBar: number;
  focusBarRange: [number, number];
  diagnosis: string;
  exerciseUrl: string;
  barTimeline: Array<{ bar: number; tSec: number }>;
  perBarScores: Record<number, BarQualityScores>;
};
