// apps/web/src/hooks/useProofCardTimeline.test.ts
import { renderHook, act } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import type { RefObject } from "react";
import type { ScoreIR } from "../lib/score-ir";

const BAR_TIMELINE = [
  { bar: 1, tSec: 0.0 },
  { bar: 2, tSec: 4.2 },
  { bar: 3, tSec: 8.5 },
  { bar: 4, tSec: 12.8 },
  { bar: 5, tSec: 17.1 },
];

// ScoreIR fixture with quaternote qstampStart values (4/4 time, 4 quarter notes per bar)
const MOCK_SCORE_IR: ScoreIR = {
  pieceId: "chopin.nocturnes.9-2",
  verovioVersion: "6.1.0",
  pageWidth: 2400,
  pages: [{ pageN: 1, viewBox: "0 0 2400 800", width: 2400, height: 800, systemBboxes: [] }],
  bars: [
    { barNumber: 1, measureOn: "m1", pageN: 1, bbox: { x: 100, y: 0, w: 0, h: 0 }, noteIds: [], qstampStart: 0, qstampEnd: 4 },
    { barNumber: 2, measureOn: "m2", pageN: 1, bbox: { x: 200, y: 0, w: 0, h: 0 }, noteIds: [], qstampStart: 4, qstampEnd: 8 },
    { barNumber: 3, measureOn: "m3", pageN: 1, bbox: { x: 300, y: 0, w: 0, h: 0 }, noteIds: [], qstampStart: 8, qstampEnd: 12 },
    { barNumber: 4, measureOn: "m4", pageN: 1, bbox: { x: 400, y: 0, w: 0, h: 0 }, noteIds: [], qstampStart: 12, qstampEnd: 16 },
    { barNumber: 5, measureOn: "m5", pageN: 1, bbox: { x: 500, y: 0, w: 0, h: 0 }, noteIds: [], qstampStart: 16, qstampEnd: 20 },
  ],
  notes: {},
};

describe("useProofCardTimeline", () => {
  it("qstampForTime returns qstampStart=0 for bar 1 at t=0.0", async () => {
    const { useProofCardTimeline } = await import("./useProofCardTimeline");
    const audioRef = { current: null } as RefObject<HTMLAudioElement | null>;
    const { result } = renderHook(() =>
      useProofCardTimeline(audioRef, MOCK_SCORE_IR, BAR_TIMELINE),
    );
    expect(result.current.qstampForTime(0.0)).toBe(0);
  });

  it("qstampForTime returns qstampStart=12 for bar 4 at t=14.0 (between bar 4 at 12.8 and bar 5 at 17.1)", async () => {
    const { useProofCardTimeline } = await import("./useProofCardTimeline");
    const audioRef = { current: null } as RefObject<HTMLAudioElement | null>;
    const { result } = renderHook(() =>
      useProofCardTimeline(audioRef, MOCK_SCORE_IR, BAR_TIMELINE),
    );
    expect(result.current.qstampForTime(14.0)).toBe(12);
  });

  it("setCurrentTime updates currentTime state", async () => {
    const { useProofCardTimeline } = await import("./useProofCardTimeline");
    const audioRef = { current: null } as RefObject<HTMLAudioElement | null>;
    const { result } = renderHook(() =>
      useProofCardTimeline(audioRef, MOCK_SCORE_IR, BAR_TIMELINE),
    );
    act(() => {
      result.current.setCurrentTime(8.5);
    });
    expect(result.current.currentTime).toBe(8.5);
  });

  it("qstampForTime returns null for empty barTimeline", async () => {
    const { useProofCardTimeline } = await import("./useProofCardTimeline");
    const audioRef = { current: null } as RefObject<HTMLAudioElement | null>;
    const { result } = renderHook(() =>
      useProofCardTimeline(audioRef, MOCK_SCORE_IR, []),
    );
    expect(result.current.qstampForTime(5.0)).toBeNull();
  });

  it("qstampForTime returns null when scoreIR is null (scoreIR not yet loaded)", async () => {
    const { useProofCardTimeline } = await import("./useProofCardTimeline");
    const audioRef = { current: null } as RefObject<HTMLAudioElement | null>;
    const { result } = renderHook(() =>
      useProofCardTimeline(audioRef, null, BAR_TIMELINE),
    );
    expect(result.current.qstampForTime(0.0)).toBeNull();
  });
});
