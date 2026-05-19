// apps/api/src/services/passage-manifest.test.ts
import { describe, expect, it } from "vitest";
import { buildPassageManifest } from "./passage-manifest";
import type { EnrichedChunk } from "../do/session-brain";

const baseUrl = "https://api.example.com";

function chunk(
  chunkIndex: number,
  barCoverage: [number, number] | null,
  alignments: Array<{ bar: number; expected_onset_ms: number }>,
): EnrichedChunk {
  return {
    chunkIndex,
    muq_scores: [0, 0, 0, 0, 0, 0],
    midi_notes: [],
    pedal_cc: [],
    alignment: alignments.map((a, i) => ({
      perf_index: i,
      score_index: i,
      expected_onset_ms: a.expected_onset_ms,
      bar: a.bar,
    })),
    bar_coverage: barCoverage,
  };
}

describe("buildPassageManifest", () => {
  it("builds a manifest covering the requested bar range", () => {
    const enrichedChunks: EnrichedChunk[] = [
      chunk(0, [1, 4], [
        { bar: 1, expected_onset_ms: 0 },
        { bar: 2, expected_onset_ms: 4000 },
        { bar: 3, expected_onset_ms: 8000 },
        { bar: 4, expected_onset_ms: 12000 },
      ]),
      chunk(1, [5, 8], [
        { bar: 5, expected_onset_ms: 1000 },
        { bar: 6, expected_onset_ms: 5000 },
        { bar: 7, expected_onset_ms: 9000 },
        { bar: 8, expected_onset_ms: 13000 },
      ]),
      chunk(2, [9, 12], [
        { bar: 9, expected_onset_ms: 2000 },
      ]),
    ];

    const result = buildPassageManifest({
      enrichedChunks,
      bars: [5, 8],
      pieceId: "chopin.ballades.1",
      sessionId: "00000000-0000-0000-0000-0000000000aa",
      baseUrl,
    });

    expect("error" in result).toBe(false);
    if ("error" in result) return;
    expect(result.pieceId).toBe("chopin.ballades.1");
    expect(result.bars).toEqual([5, 8]);
    expect(result.source).toEqual({
      kind: "session",
      sessionId: "00000000-0000-0000-0000-0000000000aa",
    });
    expect(result.chunks).toHaveLength(1);
    expect(result.chunks[0].chunkIndex).toBe(1);
    expect(result.chunks[0].url).toBe(
      `${baseUrl}/api/practice/chunk?sessionId=00000000-0000-0000-0000-0000000000aa&chunkIndex=1`,
    );
    expect(result.chunks[0].durationSec).toBe(15);
    expect(result.startOffsetSec).toBe(1.0);
    expect(result.endOffsetSec).toBe(13.0);
    expect(result.barTimeline).toEqual([
      { bar: 5, tSec: 0 },
      { bar: 6, tSec: 4 },
      { bar: 7, tSec: 8 },
      { bar: 8, tSec: 12 },
    ]);
  });

  it("returns no_alignment when no chunks have bar_coverage at all", () => {
    const enrichedChunks: EnrichedChunk[] = [
      chunk(0, null, []),
      chunk(1, null, []),
    ];

    const result = buildPassageManifest({
      enrichedChunks,
      bars: [5, 8],
      pieceId: "chopin.ballades.1",
      sessionId: "00000000-0000-0000-0000-0000000000aa",
      baseUrl,
    });

    expect("error" in result && result.error).toBe("no_alignment");
  });

  it("returns out_of_range when alignment exists but does not cover requested bars", () => {
    const enrichedChunks: EnrichedChunk[] = [
      chunk(0, [1, 3], [
        { bar: 1, expected_onset_ms: 0 },
        { bar: 2, expected_onset_ms: 5000 },
        { bar: 3, expected_onset_ms: 10000 },
      ]),
    ];

    const result = buildPassageManifest({
      enrichedChunks,
      bars: [5, 8],
      pieceId: "chopin.ballades.1",
      sessionId: "00000000-0000-0000-0000-0000000000aa",
      baseUrl,
    });

    expect("error" in result && result.error).toBe("out_of_range");
  });
});
