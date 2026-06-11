export interface LoopClockOptions {
  clipStartQ: number;
  clipEndQ: number;
  beatsPerBar: number;
  bpmAtUnity: number;
  tempoFactor: number;
}

/**
 * LoopClock maps wall-time (milliseconds) to a qstamp within [clipStartQ, clipEndQ),
 * wrapping on each pass and prefixing a one-bar count-in during which qstampNow() returns null.
 *
 * All state is pure: no DOM, no AudioContext, no requestAnimationFrame.
 */
export class LoopClock {
  private readonly clipStartQ: number;
  private readonly clipEndQ: number;
  private readonly clipRangeQ: number;
  private readonly beatsPerBar: number;
  private readonly bpmAtUnity: number;

  private startMs: number | null = null;
  private stopped = false;

  // Tempo factor and the recalibration anchor.
  private tempoFactor: number;
  // Effective "phase origin" in Q — offset added to the linear position.
  private phaseOriginQ = 0;
  // The wall-time offset (ms relative to startMs) at which the last calibration happened.
  private calibrationMs = 0;

  constructor(opts: LoopClockOptions) {
    this.clipStartQ = opts.clipStartQ;
    this.clipEndQ = opts.clipEndQ;
    this.clipRangeQ = opts.clipEndQ - opts.clipStartQ;
    if (opts.clipEndQ <= opts.clipStartQ) {
      throw new Error(`LoopClock: clipEndQ (${opts.clipEndQ}) must be greater than clipStartQ (${opts.clipStartQ})`);
    }
    this.beatsPerBar = opts.beatsPerBar;
    this.bpmAtUnity = opts.bpmAtUnity;
    this.tempoFactor = opts.tempoFactor;
  }

  /** Begin the clock. nowMs is the current wall-time (e.g. Date.now() or performance.now()). */
  start(nowMs: number): void {
    this.startMs = nowMs;
    this.stopped = false;
    this.phaseOriginQ = 0;
    this.calibrationMs = 0;
  }

  /** Recalibrate tempo factor at the given wall-time, preserving current position. */
  setTempoFactor(factor: number, nowMs: number): void {
    if (this.startMs === null || this.stopped) {
      this.tempoFactor = factor;
      return;
    }
    // Capture current Q position before changing factor.
    const wrappedQ = this.qstampNow(nowMs);
    if (wrappedQ === null) {
      // Still in count-in; only update the factor.
      this.tempoFactor = factor;
      return;
    }
    // Record the calibration point.
    this.calibrationMs = nowMs - this.startMs;
    // phaseOriginQ is the Q offset from clipStartQ at the calibration point.
    this.phaseOriginQ = wrappedQ - this.clipStartQ;
    this.tempoFactor = factor;
  }

  /** Returns the current qstamp, or null if in count-in or stopped/not started. */
  qstampNow(nowMs: number): number | null {
    if (this.startMs === null || this.stopped) return null;

    const countInMs = this.countInMs();
    const elapsedMs = nowMs - this.startMs;
    if (elapsedMs < countInMs) return null;

    const rawQ = this.rawQAtMs(nowMs);
    // Wrap into [clipStartQ, clipEndQ).
    const range = this.clipRangeQ;
    const offset = ((rawQ - this.clipStartQ) % range + range) % range;
    return this.clipStartQ + offset;
  }

  /** Stop the clock; qstampNow returns null after this. */
  stop(): void {
    this.stopped = true;
  }

  // Q per second at current tempo factor.
  private qPerSecond(): number {
    return (this.bpmAtUnity * this.tempoFactor) / 60;
  }

  // Count-in duration in milliseconds (one bar at current tempo).
  private countInMs(): number {
    const secondsPerBeat = 60 / (this.bpmAtUnity * this.tempoFactor);
    return this.beatsPerBar * secondsPerBeat * 1000;
  }

  // Raw Q position (not wrapped) at the given wall-time.
  private rawQAtMs(nowMs: number): number {
    const countInSec = this.countInMs() / 1000;
    const elapsedSec = (nowMs - this.startMs!) / 1000 - this.calibrationMs / 1000;
    // After calibration, the count-in has already elapsed (calibrationMs >= countInMs on
    // any setTempoFactor call that happens post-count-in). On the initial pass
    // (calibrationMs = 0), subtract countInSec to anchor Q=clipStartQ at count-in end.
    const playbackSec = this.calibrationMs === 0
      ? Math.max(0, elapsedSec - countInSec)
      : elapsedSec;
    return this.clipStartQ + this.phaseOriginQ + playbackSec * this.qPerSecond();
  }
}
