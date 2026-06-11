// apps/web/src/hooks/useLoopPlayer.test.ts
import { renderHook, act } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import type { ScoreIR } from "../lib/score-ir";
import type { ClipNote } from "../lib/score-worker";

const STUB_IR: ScoreIR = {
  pieceId: "test",
  verovioVersion: "4.0.0",
  pageWidth: 1600,
  pages: [{ pageN: 1, viewBox: "0 0 1600 600", width: 1600, height: 600, systemBboxes: [] }],
  bars: [
    { barNumber: 1, measureOn: "m1", pageN: 1, bbox: { x: 0, y: 0, w: 0, h: 0 }, noteIds: [], qstampStart: 0, qstampEnd: 4 },
  ],
  notes: {},
};

const STUB_NOTES: ClipNote[] = [{ midi: 60, startQ: 0, endQ: 2 }];

// State shared between the mock factory (hoisted) and the test body.
// Cannot reference top-level `let` from inside a vi.mock factory due to hoisting,
// so we use a plain object to hold mutable state.
const shared = {
  playerState: "idle" as string,
  playFn: null as null | ReturnType<typeof vi.fn>,
  destroyFn: null as null | ReturnType<typeof vi.fn>,
};

vi.mock("../lib/loop-player", () => {
  const playFn = vi.fn().mockImplementation(async () => {
    shared.playerState = "counting-in";
  });
  const destroyFn = vi.fn();
  shared.playFn = playFn;
  shared.destroyFn = destroyFn;

  class MockLoopPlayer {
    get state() { return shared.playerState; }
    audioUnavailable = false;
    tempoFactor = 1.0;
    play = playFn;
    pause = vi.fn();
    stop = vi.fn();
    destroy = destroyFn;
    setTempoFactor = vi.fn();
    qstampSource = vi.fn().mockReturnValue(null);
  }
  return { LoopPlayer: MockLoopPlayer };
});

// AudioContext as a real class so `new AudioContext()` works.
const mockAudioContextClose = vi.fn().mockResolvedValue(undefined);
class MockAudioContext {
  state = "running";
  currentTime = 0;
  close = mockAudioContextClose;
}

describe("useLoopPlayer", () => {
  beforeEach(() => {
    vi.useFakeTimers();
    shared.playerState = "idle";
    vi.clearAllMocks();
    vi.stubGlobal("AudioContext", MockAudioContext);
  });

  afterEach(() => {
    vi.useRealTimers();
    vi.unstubAllGlobals();
  });

  it("count-in watcher interval is cleared on unmount — no setIsCounting calls after unmount", async () => {
    const { useLoopPlayer } = await import("./useLoopPlayer");

    const { result, unmount } = renderHook(() =>
      useLoopPlayer({
        clipIR: STUB_IR,
        clipNotes: STUB_NOTES,
        beatsPerBar: 4,
        bpmAtUnity: 120,
        tempoFactor: 1.0,
      })
    );

    // Trigger play — mock puts player into counting-in state.
    await act(async () => {
      result.current.play();
      // Flush the play().then() microtask
      await Promise.resolve();
      await Promise.resolve();
    });

    // isCounting should be true now (player.state === "counting-in").
    expect(result.current.isCounting).toBe(true);

    // Spy on clearInterval before unmount.
    const clearIntervalSpy = vi.spyOn(globalThis, "clearInterval");

    // Unmount while the interval is still live (player still in counting-in state).
    unmount();

    // clearInterval must have been called (watcher cleanup on unmount).
    expect(clearIntervalSpy).toHaveBeenCalled();

    // Advance fake timers well past several watcher ticks (50ms each).
    // If the interval were still running it would attempt state updates.
    vi.advanceTimersByTime(500);

    // destroy() must have been called as part of unmount cleanup.
    expect(shared.destroyFn).toHaveBeenCalled();

    clearIntervalSpy.mockRestore();
  });
});
