// apps/web/src/lib/passage-player.test.ts
import { beforeEach, describe, expect, it, vi } from "vitest";
import { PassagePlayer } from "./passage-player";

type PassageManifest = {
  source: { kind: "session"; sessionId: string };
  pieceId: string;
  bars: [number, number];
  chunks: Array<{ url: string; chunkIndex: number; durationSec: number }>;
  startOffsetSec: number;
  endOffsetSec: number;
  barTimeline: Array<{ bar: number; tSec: number }>;
};

const manifest: PassageManifest = {
  source: { kind: "session", sessionId: "s1" },
  pieceId: "chopin.ballades.1",
  bars: [5, 8],
  chunks: [{ url: "https://api/c1.webm", chunkIndex: 1, durationSec: 15 }],
  startOffsetSec: 1.0,
  endOffsetSec: 13.0,
  barTimeline: [
    { bar: 5, tSec: 0 },
    { bar: 6, tSec: 4 },
    { bar: 7, tSec: 8 },
    { bar: 8, tSec: 12 },
  ],
};

function makeStubAudioContext() {
  const decodedBuffer = { duration: 15, length: 15 * 44100 } as AudioBuffer;
  const startCalls: Array<{ when: number; offset: number; duration?: number }> = [];
  const ctx = {
    currentTime: 0,
    state: "running",
    decodeAudioData: vi.fn().mockResolvedValue(decodedBuffer),
    createBufferSource: vi.fn(() => ({
      buffer: null as AudioBuffer | null,
      connect: vi.fn(),
      start: vi.fn((when?: number, offset?: number, duration?: number) => {
        startCalls.push({ when: when ?? 0, offset: offset ?? 0, duration });
      }),
      stop: vi.fn(),
      onended: null as (() => void) | null,
    })),
    destination: {},
    resume: vi.fn().mockResolvedValue(undefined),
    close: vi.fn().mockResolvedValue(undefined),
  } as unknown as AudioContext;
  return { ctx, startCalls };
}

beforeEach(() => {
  globalThis.fetch = vi
    .fn()
    .mockResolvedValue(
      new Response(new ArrayBuffer(1024), { status: 200 }),
    ) as typeof fetch;
});

describe("PassagePlayer", () => {
  it("transitions to ready and exposes duration after load", async () => {
    const { ctx } = makeStubAudioContext();
    const player = new PassagePlayer(manifest, ctx);
    expect(player.state).toBe("idle");
    await player.load();
    expect(player.state).toBe("ready");
    expect(player.duration).toBe(13);
  });

  it("play() schedules source with startOffsetSec and emits monotonic ticks", async () => {
    const { ctx, startCalls } = makeStubAudioContext();
    const player = new PassagePlayer(manifest, ctx);
    await player.load();

    const ticks: number[] = [];
    player.onTick((t) => ticks.push(t));

    player.play();
    expect(player.state).toBe("playing");
    expect(startCalls).toHaveLength(1);
    expect(startCalls[0].offset).toBe(1.0);

    (ctx as unknown as { currentTime: number }).currentTime = 0.05;
    await player.__testTick();
    (ctx as unknown as { currentTime: number }).currentTime = 0.1;
    await player.__testTick();
    (ctx as unknown as { currentTime: number }).currentTime = 0.2;
    await player.__testTick();

    expect(ticks.length).toBeGreaterThanOrEqual(3);
    for (let i = 1; i < ticks.length; i++) {
      expect(ticks[i]).toBeGreaterThan(ticks[i - 1]);
    }
  });

  it("pause() stops scheduled sources and transitions to paused", async () => {
    const { ctx } = makeStubAudioContext();
    const player = new PassagePlayer(manifest, ctx);
    await player.load();
    player.play();
    expect(player.state).toBe("playing");

    player.pause();
    expect(player.state).toBe("paused");

    type SourceMock = { stop: { mock: { calls: unknown[] } } };
    type CtxWithMock = AudioContext & {
      createBufferSource: { mock: { results: Array<{ value: SourceMock }> } };
    };

    const results = (ctx as CtxWithMock).createBufferSource.mock.results;
    const totalStopCalls = results.reduce(
      (acc, r) => acc + r.value.stop.mock.calls.length,
      0,
    );
    expect(totalStopCalls).toBeGreaterThanOrEqual(1);
  });
});
