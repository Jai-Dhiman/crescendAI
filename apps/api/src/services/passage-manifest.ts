// apps/api/src/services/passage-manifest.ts
import type { EnrichedChunk } from "../do/session-brain";

const CHUNK_DURATION_SEC = 15;

export interface PassageManifest {
  source: { kind: "session"; sessionId: string };
  pieceId: string;
  bars: [number, number];
  chunks: Array<{ url: string; chunkIndex: number; durationSec: number }>;
  startOffsetSec: number;
  endOffsetSec: number;
  barTimeline: Array<{ bar: number; tSec: number }>;
}

export type PassageManifestError = { error: "no_alignment" | "out_of_range" };

export interface BuildPassageManifestArgs {
  enrichedChunks: EnrichedChunk[];
  bars: [number, number];
  pieceId: string;
  sessionId: string;
  baseUrl: string;
}

export function buildPassageManifest(
  args: BuildPassageManifestArgs,
): PassageManifest | PassageManifestError {
  const [startBar, endBar] = args.bars;

  const covering = args.enrichedChunks.filter(
    (c) =>
      c.bar_coverage !== null &&
      c.bar_coverage[1] >= startBar &&
      c.bar_coverage[0] <= endBar,
  );

  const anyAlignedChunk = args.enrichedChunks.some(
    (c) => c.bar_coverage !== null,
  );
  if (!anyAlignedChunk) {
    return { error: "no_alignment" };
  }

  if (covering.length === 0) {
    return { error: "out_of_range" };
  }

  covering.sort((a, b) => a.chunkIndex - b.chunkIndex);
  const firstChunkIndex = covering[0].chunkIndex;

  const barTimeline: Array<{ bar: number; tSec: number }> = [];

  for (let bar = startBar; bar <= endBar; bar++) {
    for (const c of covering) {
      const hit = c.alignment.find((a) => a.bar === bar);
      if (hit !== undefined) {
        const tSec =
          (c.chunkIndex - firstChunkIndex) * CHUNK_DURATION_SEC +
          hit.expected_onset_ms / 1000;
        barTimeline.push({ bar, tSec });
        break;
      }
    }
  }

  if (barTimeline.length === 0) {
    return { error: "no_alignment" };
  }

  const startOffsetSec = barTimeline[0].tSec;
  const lastBarHit = barTimeline[barTimeline.length - 1];
  const endOffsetSec = lastBarHit.tSec;
  for (const entry of barTimeline) {
    entry.tSec -= startOffsetSec;
  }

  return {
    source: { kind: "session", sessionId: args.sessionId },
    pieceId: args.pieceId,
    bars: args.bars,
    chunks: covering.map((c) => ({
      url: `${args.baseUrl}/api/practice/chunk?sessionId=${args.sessionId}&chunkIndex=${c.chunkIndex}`,
      chunkIndex: c.chunkIndex,
      durationSec: CHUNK_DURATION_SEC,
    })),
    startOffsetSec,
    endOffsetSec,
    barTimeline,
  };
}
