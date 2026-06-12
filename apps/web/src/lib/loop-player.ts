import { Soundfont } from "smplr";
import { LoopClock } from "./loop-clock";
import type { ClipNote } from "./score-worker";
import type { ScoreIR } from "./score-ir";
import { Sentry } from "./sentry";

export type LoopPlayerState = "idle" | "counting-in" | "playing" | "paused";

export interface LoopPlayerOptions {
  ctx: AudioContext;
  instrumentUrl: string;
  clipIR: ScoreIR;
  clipNotes: ClipNote[];
  beatsPerBar: number;
  bpmAtUnity: number;
  tempoFactor: number;
}

const LOOKAHEAD_SEC = 0.1;
const SCHEDULER_INTERVAL_MS = 25;

export class LoopPlayer {
  state: LoopPlayerState = "idle";
  tempoFactor: number;
  audioUnavailable = false;

  private readonly ctx: AudioContext;
  private readonly clipNotes: ClipNote[];
  private readonly beatsPerBar: number;
  private readonly bpmAtUnity: number;
  private readonly clipStartQ: number;
  private readonly clipEndQ: number;

  private piano: ReturnType<typeof Soundfont> | null = null;
  private pianoLoadPromise: Promise<void> | null = null;
  private clock: LoopClock | null = null;
  private schedulerTimer: ReturnType<typeof setInterval> | null = null;
  private scheduledNoteKeys = new Set<string>();
  private currentPassIndex = 0;

  // Metronome state — initialized in play(), reset in stop()
  private nextMetronomeBeatTime: number | null = null;
  private metronomeBeatIndex = 0;

  constructor(opts: LoopPlayerOptions) {
    this.ctx = opts.ctx;
    this.tempoFactor = opts.tempoFactor;
    this.clipNotes = opts.clipNotes;
    this.beatsPerBar = opts.beatsPerBar;
    this.bpmAtUnity = opts.bpmAtUnity;

    const bars = opts.clipIR.bars;
    this.clipStartQ = bars.length > 0 ? bars[0].qstampStart : 0;
    this.clipEndQ = bars.length > 0 ? bars[bars.length - 1].qstampEnd : 0;

    this.pianoLoadPromise = this.loadInstrument(opts.instrumentUrl);
  }

  private async loadInstrument(instrumentUrl: string): Promise<void> {
    try {
      // Pass the self-hosted file via `instrumentUrl` (a direct URL), NOT
      // `instrument` (an instrument NAME, which makes smplr build a MusyngKite
      // CDN URL and ignore our self-hosted samples). smplr's SoundfontConfig is
      // { kit, instrument?, instrumentUrl } — instrumentUrl wins for self-hosting.
      this.piano = Soundfont(this.ctx, { instrumentUrl });
      // .load is a Promise property on the Smplr interface (not a method call)
      await this.piano.load;
    } catch (err) {
      this.audioUnavailable = true;
      Sentry.captureException(err);
      console.error("[LoopPlayer] smplr Soundfont load failed:", err);
    }
  }

  async play(): Promise<void> {
    if (this.state === "playing" || this.state === "counting-in") return;

    if (this.ctx.state === "suspended") {
      await this.ctx.resume();
    }

    if (this.pianoLoadPromise) {
      await this.pianoLoadPromise;
    }

    const nowMs = performance.now();
    this.clock = new LoopClock({
      clipStartQ: this.clipStartQ,
      clipEndQ: this.clipEndQ,
      beatsPerBar: this.beatsPerBar,
      bpmAtUnity: this.bpmAtUnity,
      tempoFactor: this.tempoFactor,
    });
    this.clock.start(nowMs);
    this.scheduledNoteKeys.clear();
    this.currentPassIndex = 0;
    // Initialize metronome for the first scheduler tick
    this.nextMetronomeBeatTime = this.ctx.currentTime;
    this.metronomeBeatIndex = 0;

    this.state = "counting-in";
    this.startScheduler();
  }

  pause(): void {
    if (this.state !== "playing" && this.state !== "counting-in") return;
    this.stopScheduler();
    this.clock?.stop();
    this.state = "paused";
  }

  stop(): void {
    this.stopScheduler();
    this.clock?.stop();
    this.clock = null;
    this.scheduledNoteKeys.clear();
    // Reset metronome state so stop-then-replay starts fresh
    this.nextMetronomeBeatTime = null;
    this.metronomeBeatIndex = 0;
    this.state = "idle";
  }

  setTempoFactor(factor: number): void {
    this.tempoFactor = factor;
    this.clock?.setTempoFactor(factor, performance.now());
  }

  qstampSource(): number | null {
    if (!this.clock) return null;
    return this.clock.qstampNow(performance.now());
  }

  destroy(): void {
    this.stop();
  }

  private startScheduler(): void {
    if (this.schedulerTimer !== null) return;
    this.schedulerTimer = setInterval(() => this.scheduleTick(), SCHEDULER_INTERVAL_MS);
  }

  private stopScheduler(): void {
    if (this.schedulerTimer !== null) {
      clearInterval(this.schedulerTimer);
      this.schedulerTimer = null;
    }
  }

  private scheduleTick(): void {
    if (!this.clock) return;

    const nowMs = performance.now();
    const nowQ = this.clock.qstampNow(nowMs);

    if (nowQ === null) {
      this.scheduleMetronome();
      return;
    }

    if (this.state === "counting-in") {
      this.state = "playing";
    }

    const qPerSec = (this.bpmAtUnity * this.tempoFactor) / 60;
    const lookaheadQ = LOOKAHEAD_SEC * qPerSec;
    const horizonQ = nowQ + lookaheadQ;

    // Pass-wrap detection: uses floor division to fire exactly ONCE per loop boundary.
    // horizonQ may exceed clipEndQ; passOfHorizon advances only when floor crosses a new integer.
    const clipRange = this.clipEndQ - this.clipStartQ;
    const passOfHorizon = Math.floor((horizonQ - this.clipStartQ) / clipRange);
    const wrapsThisTick = passOfHorizon > this.currentPassIndex;
    if (wrapsThisTick) {
      this.currentPassIndex = passOfHorizon;
      this.scheduledNoteKeys.clear();
    }

    for (let i = 0; i < this.clipNotes.length; i++) {
      const note = this.clipNotes[i];
      const noteQ = note.startQ;

      // Case A: note is in the tail of the current pass
      const inCurrentPassTail = noteQ >= nowQ && noteQ < this.clipEndQ;
      // Case B: wrap occurred this tick AND note is in the head of the next pass
      const inNextPassHead = wrapsThisTick && noteQ >= this.clipStartQ && noteQ < (horizonQ - clipRange);

      if (!inCurrentPassTail && !inNextPassHead) continue;

      const key = `${this.currentPassIndex}-${i}`;
      if (this.scheduledNoteKeys.has(key)) continue;
      this.scheduledNoteKeys.add(key);

      if (this.piano && !this.audioUnavailable) {
        const effectiveNoteQ = inNextPassHead ? noteQ + clipRange : noteQ;
        const qOffset = effectiveNoteQ - nowQ;
        const secOffset = qOffset / qPerSec;
        const audioTime = this.ctx.currentTime + secOffset;
        const durationQ = note.endQ - note.startQ;
        const durationSec = Math.max(0.05, durationQ / qPerSec);

        this.piano.start({
          note: note.midi,
          time: audioTime,
          duration: durationSec,
          velocity: 80,
        });
      }
    }

    this.scheduleMetronome();
  }

  private scheduleMetronome(): void {
    const secPerBeat = 60 / (this.bpmAtUnity * this.tempoFactor);
    const lookaheadAudioTime = this.ctx.currentTime + LOOKAHEAD_SEC;

    if (this.nextMetronomeBeatTime === null) return;

    while (this.nextMetronomeBeatTime <= lookaheadAudioTime) {
      this.playMetronomeClick(
        this.nextMetronomeBeatTime,
        this.metronomeBeatIndex % this.beatsPerBar === 0,
      );
      this.nextMetronomeBeatTime += secPerBeat;
      this.metronomeBeatIndex++;
    }
  }

  private playMetronomeClick(time: number, accent: boolean): void {
    const osc = this.ctx.createOscillator();
    const gain = this.ctx.createGain();
    osc.connect(gain);
    gain.connect(this.ctx.destination);
    osc.frequency.value = accent ? 1000 : 800;
    osc.type = "sine";
    gain.gain.setValueAtTime(accent ? 0.5 : 0.25, time);
    gain.gain.exponentialRampToValueAtTime(0.001, time + 0.04);
    osc.start(time);
    osc.stop(time + 0.04);
  }
}
