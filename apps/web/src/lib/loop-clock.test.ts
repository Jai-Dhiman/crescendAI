import { describe, expect, it } from "vitest";
import { LoopClock } from "./loop-clock";

const OPTS = {
  clipStartQ: 16,   // bar 5 in a 4/4 piece at 4Q/bar
  clipEndQ: 32,     // bar 9 (exclusive)
  beatsPerBar: 4,
  bpmAtUnity: 120,  // 120 BPM at 1.0x
  tempoFactor: 1.0,
};

describe("LoopClock", () => {
  it("returns null before start() is called", () => {
    const clock = new LoopClock(OPTS);
    expect(clock.qstampNow(1000)).toBeNull();
  });

  it("returns null during count-in (first bar of metronome)", () => {
    const clock = new LoopClock(OPTS);
    clock.start(0); // startMs = 0
    // Count-in = 1 bar = beatsPerBar / (bpmAtUnity * tempoFactor) * 60 s
    // = 4 / (120 * 1.0) * 60 = 2 seconds = 2000ms
    // At 1000ms we are still in count-in.
    expect(clock.qstampNow(1000)).toBeNull();
  });

  it("returns clipStartQ immediately after count-in ends", () => {
    const clock = new LoopClock(OPTS);
    clock.start(0);
    // Count-in = 2000ms at 1.0x tempo.
    // At exactly 2000ms: elapsed = 0Q past clipStart.
    const q = clock.qstampNow(2000);
    expect(q).not.toBeNull();
    expect(q!).toBeCloseTo(OPTS.clipStartQ, 2);
  });

  it("advances qstamp proportionally to elapsed time", () => {
    const clock = new LoopClock(OPTS);
    clock.start(0);
    // After count-in (2000ms), one full clip = (clipEndQ - clipStartQ) / (bpmAtUnity * tempoFactor / 60)
    // = 16Q / (120/60 Q/s) = 16 / 2 = 8 seconds = 8000ms.
    // At 2000ms (count-in end) + 4000ms (half-clip) = 6000ms:
    const q = clock.qstampNow(6000);
    expect(q).not.toBeNull();
    // Half of clip range: clipStartQ + 8 = 24
    expect(q!).toBeCloseTo(24, 1);
  });

  it("wraps back to clipStartQ when clip duration elapses", () => {
    const clock = new LoopClock(OPTS);
    clock.start(0);
    // count-in (2000ms) + full clip (8000ms) = 10000ms → start of second pass
    const q = clock.qstampNow(10000);
    expect(q).not.toBeNull();
    expect(q!).toBeCloseTo(OPTS.clipStartQ, 2);
  });

  it("scales elapsed time by tempoFactor (0.5x takes twice as long)", () => {
    const clock = new LoopClock({ ...OPTS, tempoFactor: 0.5 });
    clock.start(0);
    // count-in at 0.5x = 4 seconds = 4000ms.
    // Still in count-in at 2000ms.
    expect(clock.qstampNow(2000)).toBeNull();
    // At 4000ms: just past count-in, q ≈ clipStartQ.
    const q = clock.qstampNow(4000);
    expect(q).not.toBeNull();
    expect(q!).toBeCloseTo(OPTS.clipStartQ, 2);
  });

  it("setTempoFactor rescales future qstamps without resetting position", () => {
    const clock = new LoopClock(OPTS);
    clock.start(0);
    // count-in = 2000ms. At t=6000ms: playback elapsed = 4s × 2 Q/s = 8Q → q = 16+8 = 24.
    const q1 = clock.qstampNow(6000);
    expect(q1).not.toBeNull();
    expect(q1!).toBeCloseTo(24, 1);

    // Halve tempo at t=6000ms — recalibrate so current position (q=24) is preserved.
    clock.setTempoFactor(0.5, 6000);

    // At t=6000ms + 1000ms = 7000ms: with 0.5x, qPerSec = 120*0.5/60 = 1 Q/s.
    // 1 second elapsed since calibration at q=24 → q ≈ 25.
    const q2 = clock.qstampNow(7000);
    expect(q2).not.toBeNull();
    expect(q2!).toBeCloseTo(25, 1);
  });

  it("stop() causes qstampNow to return null", () => {
    const clock = new LoopClock(OPTS);
    clock.start(0);
    clock.stop();
    expect(clock.qstampNow(3000)).toBeNull();
  });
});
