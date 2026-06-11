import { beforeEach, describe, expect, it, vi } from "vitest";
import type { ClipNote } from "./score-worker";
import type { ScoreIR } from "./score-ir";

// Minimal stub ScoreIR
const STUB_IR: ScoreIR = {
  pieceId: "test",
  verovioVersion: "4.0.0",
  pageWidth: 1600,
  pages: [{ pageN: 1, viewBox: "0 0 1600 600", width: 1600, height: 600, systemBboxes: [] }],
  bars: [
    { barNumber: 5, measureOn: "m5", pageN: 1, bbox: { x: 0, y: 0, w: 0, h: 0 }, noteIds: [], qstampStart: 16, qstampEnd: 20 },
    { barNumber: 6, measureOn: "m6", pageN: 1, bbox: { x: 100, y: 0, w: 0, h: 0 }, noteIds: [], qstampStart: 20, qstampEnd: 24 },
    { barNumber: 7, measureOn: "m7", pageN: 1, bbox: { x: 200, y: 0, w: 0, h: 0 }, noteIds: [], qstampStart: 24, qstampEnd: 28 },
    { barNumber: 8, measureOn: "m8", pageN: 1, bbox: { x: 300, y: 0, w: 0, h: 0 }, noteIds: [], qstampStart: 28, qstampEnd: 32 },
  ],
  notes: {},
};

const STUB_NOTES: ClipNote[] = [
  { midi: 60, startQ: 16, endQ: 18 },
  { midi: 64, startQ: 20, endQ: 22 },
];

vi.mock("smplr", () => ({
  Soundfont: vi.fn().mockImplementation((_ctx: unknown, _opts: unknown) => ({
    load: vi.fn().mockResolvedValue(undefined),
    start: vi.fn(),
    stop: vi.fn(),
    loaded: true,
  })),
}));

describe("LoopPlayer", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    vi.useFakeTimers();
  });

  it("starts in idle state", async () => {
    const { LoopPlayer } = await import("./loop-player");
    const ctx = { currentTime: 0, state: "running", destination: {}, resume: vi.fn().mockResolvedValue(undefined), close: vi.fn().mockResolvedValue(undefined), createOscillator: vi.fn(() => ({ frequency: { value: 0 }, type: "sine", start: vi.fn(), stop: vi.fn(), connect: vi.fn() })), createGain: vi.fn(() => ({ gain: { setValueAtTime: vi.fn(), exponentialRampToValueAtTime: vi.fn() }, connect: vi.fn() })) } as unknown as AudioContext;
    const player = new LoopPlayer({
      ctx,
      instrumentUrl: "/soundfonts/acoustic_grand_piano-mp3.js",
      clipIR: STUB_IR,
      clipNotes: STUB_NOTES,
      beatsPerBar: 4,
      bpmAtUnity: 120,
      tempoFactor: 1.0,
    });
    expect(player.state).toBe("idle");
    player.destroy();
  });

  it("transitions to counting-in then playing after play()", async () => {
    const { LoopPlayer } = await import("./loop-player");
    const ctx = { currentTime: 0, state: "running", destination: {}, resume: vi.fn().mockResolvedValue(undefined), close: vi.fn().mockResolvedValue(undefined), createOscillator: vi.fn(() => ({ frequency: { value: 0 }, type: "sine", start: vi.fn(), stop: vi.fn(), connect: vi.fn() })), createGain: vi.fn(() => ({ gain: { setValueAtTime: vi.fn(), exponentialRampToValueAtTime: vi.fn() }, connect: vi.fn() })) } as unknown as AudioContext;
    const player = new LoopPlayer({
      ctx,
      instrumentUrl: "/soundfonts/acoustic_grand_piano-mp3.js",
      clipIR: STUB_IR,
      clipNotes: STUB_NOTES,
      beatsPerBar: 4,
      bpmAtUnity: 120,
      tempoFactor: 1.0,
    });
    await player.play();
    expect(player.state === "counting-in" || player.state === "playing").toBe(true);
    player.destroy();
  });

  it("pause() transitions from playing to paused", async () => {
    const { LoopPlayer } = await import("./loop-player");
    const ctx = { currentTime: 0, state: "running", destination: {}, resume: vi.fn().mockResolvedValue(undefined), close: vi.fn().mockResolvedValue(undefined), createOscillator: vi.fn(() => ({ frequency: { value: 0 }, type: "sine", start: vi.fn(), stop: vi.fn(), connect: vi.fn() })), createGain: vi.fn(() => ({ gain: { setValueAtTime: vi.fn(), exponentialRampToValueAtTime: vi.fn() }, connect: vi.fn() })) } as unknown as AudioContext;
    const player = new LoopPlayer({
      ctx,
      instrumentUrl: "/soundfonts/acoustic_grand_piano-mp3.js",
      clipIR: STUB_IR,
      clipNotes: STUB_NOTES,
      beatsPerBar: 4,
      bpmAtUnity: 120,
      tempoFactor: 1.0,
    });
    await player.play();
    player.pause();
    expect(player.state).toBe("paused");
    player.destroy();
  });

  it("stop() transitions to idle and qstampSource returns null", async () => {
    const { LoopPlayer } = await import("./loop-player");
    const ctx = { currentTime: 0, state: "running", destination: {}, resume: vi.fn().mockResolvedValue(undefined), close: vi.fn().mockResolvedValue(undefined), createOscillator: vi.fn(() => ({ frequency: { value: 0 }, type: "sine", start: vi.fn(), stop: vi.fn(), connect: vi.fn() })), createGain: vi.fn(() => ({ gain: { setValueAtTime: vi.fn(), exponentialRampToValueAtTime: vi.fn() }, connect: vi.fn() })) } as unknown as AudioContext;
    const player = new LoopPlayer({
      ctx,
      instrumentUrl: "/soundfonts/acoustic_grand_piano-mp3.js",
      clipIR: STUB_IR,
      clipNotes: STUB_NOTES,
      beatsPerBar: 4,
      bpmAtUnity: 120,
      tempoFactor: 1.0,
    });
    await player.play();
    player.stop();
    expect(player.state).toBe("idle");
    expect(player.qstampSource()).toBeNull();
    player.destroy();
  });

  it("setTempoFactor updates tempoFactor on the clock", async () => {
    const { LoopPlayer } = await import("./loop-player");
    const ctx = { currentTime: 0, state: "running", destination: {}, resume: vi.fn().mockResolvedValue(undefined), close: vi.fn().mockResolvedValue(undefined), createOscillator: vi.fn(() => ({ frequency: { value: 0 }, type: "sine", start: vi.fn(), stop: vi.fn(), connect: vi.fn() })), createGain: vi.fn(() => ({ gain: { setValueAtTime: vi.fn(), exponentialRampToValueAtTime: vi.fn() }, connect: vi.fn() })) } as unknown as AudioContext;
    const player = new LoopPlayer({
      ctx,
      instrumentUrl: "/soundfonts/acoustic_grand_piano-mp3.js",
      clipIR: STUB_IR,
      clipNotes: STUB_NOTES,
      beatsPerBar: 4,
      bpmAtUnity: 120,
      tempoFactor: 1.0,
    });
    await player.play();
    player.setTempoFactor(0.5);
    expect(player.tempoFactor).toBe(0.5);
    player.destroy();
  });

  it("first note of pass N+1 is scheduled in the wrap-boundary tick (no dropout, no doubling)", async () => {
    const { LoopPlayer } = await import("./loop-player");
    const ctx = { currentTime: 0, state: "running", destination: {}, resume: vi.fn().mockResolvedValue(undefined), close: vi.fn().mockResolvedValue(undefined), createOscillator: vi.fn(() => ({ frequency: { value: 0 }, type: "sine", start: vi.fn(), stop: vi.fn(), connect: vi.fn() })), createGain: vi.fn(() => ({ gain: { setValueAtTime: vi.fn(), exponentialRampToValueAtTime: vi.fn() }, connect: vi.fn() })) } as unknown as AudioContext;

    const SINGLE_NOTE: ClipNote[] = [
      { midi: 60, startQ: 16, endQ: 18 },
    ];
    const player = new LoopPlayer({
      ctx,
      instrumentUrl: "/soundfonts/acoustic_grand_piano-mp3.js",
      clipIR: STUB_IR,
      clipNotes: SINGLE_NOTE,
      beatsPerBar: 4,
      bpmAtUnity: 120,
      tempoFactor: 1.0,
    });
    await player.play();

    // Advance fake timers by count-in (2s) + full clip (8s) + several scheduler ticks.
    vi.advanceTimersByTime(10200);

    const { Soundfont } = await import("smplr");
    const sfInstance = (Soundfont as ReturnType<typeof vi.fn>).mock.results[0]?.value;
    expect(sfInstance).toBeDefined();
    const startCalls = (sfInstance.start as ReturnType<typeof vi.fn>).mock.calls as Array<[{ note: number }]>;
    const note60Calls = startCalls.filter((args) => args[0]?.note === 60);
    // Should be exactly 2: pass 1 and pass 2 (wrap). Doubling bug = 3+, dropout = 1.
    expect(note60Calls.length).toBe(2);
    player.destroy();
  });
});
