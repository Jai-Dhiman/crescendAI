// apps/web/src/lib/passage-player.ts
type PassageManifest = {
  source: { kind: "session"; sessionId: string };
  pieceId: string;
  bars: [number, number];
  chunks: Array<{ url: string; chunkIndex: number; durationSec: number }>;
  startOffsetSec: number;
  endOffsetSec: number;
  barTimeline: Array<{ bar: number; tSec: number }>;
};

type PlayerState = "idle" | "loading" | "ready" | "playing" | "paused" | "ended" | "error";

type TickCb = (tSec: number) => void;

export class PassagePlayer {
  state: PlayerState = "idle";
  duration = 0;
  private buffers: AudioBuffer[] = [];
  private manifest: PassageManifest;
  private ctx: AudioContext;
  private sources: AudioBufferSourceNode[] = [];
  private tickCallbacks = new Set<TickCb>();
  private playStartedAtCtxTime = 0;

  constructor(manifest: PassageManifest, ctx: AudioContext) {
    this.manifest = manifest;
    this.ctx = ctx;
  }

  async load(): Promise<void> {
    this.state = "loading";
    try {
      const responses = await Promise.all(
        this.manifest.chunks.map((c) => fetch(c.url, { credentials: "include" })),
      );
      const arrayBuffers = await Promise.all(responses.map((r) => r.arrayBuffer()));
      this.buffers = await Promise.all(
        arrayBuffers.map((ab) => this.ctx.decodeAudioData(ab)),
      );
      this.duration = this.buffers.reduce((acc, buf, i) => {
        const offset = i === 0 ? this.manifest.startOffsetSec : 0;
        const remaining =
          i === this.buffers.length - 1
            ? this.manifest.endOffsetSec - offset
            : buf.duration - offset;
        return acc + remaining;
      }, 0);
      this.state = "ready";
    } catch (err) {
      this.state = "error";
      throw err;
    }
  }

  onTick(cb: TickCb): () => void {
    this.tickCallbacks.add(cb);
    return () => this.tickCallbacks.delete(cb);
  }

  play(): void {
    if (this.state !== "ready" && this.state !== "paused") return;
    this.playStartedAtCtxTime = this.ctx.currentTime;
    const baseWhen = this.ctx.currentTime;
    let cursor = 0;
    this.sources = [];
    for (let i = 0; i < this.buffers.length; i++) {
      const buf = this.buffers[i];
      const source = this.ctx.createBufferSource();
      source.buffer = buf;
      source.connect(this.ctx.destination);
      const offset = i === 0 ? this.manifest.startOffsetSec : 0;
      const remaining =
        i === this.buffers.length - 1
          ? this.manifest.endOffsetSec - offset
          : buf.duration - offset;
      source.start(baseWhen + cursor, offset, remaining);
      this.sources.push(source);
      cursor += remaining;
    }
    this.state = "playing";
    this.startRafLoop();
  }

  private startRafLoop(): void {
    const tick = () => {
      if (this.state !== "playing") return;
      const t = this.ctx.currentTime - this.playStartedAtCtxTime;
      for (const cb of this.tickCallbacks) cb(t);
      if (typeof requestAnimationFrame !== "undefined") {
        requestAnimationFrame(tick);
      }
    };
    if (typeof requestAnimationFrame !== "undefined") {
      requestAnimationFrame(tick);
    }
  }

  async __testTick(): Promise<void> {
    const t = this.ctx.currentTime - this.playStartedAtCtxTime;
    for (const cb of this.tickCallbacks) cb(t);
  }

  pause(): void {
    if (this.state !== "playing") return;
    for (const s of this.sources) {
      try { s.stop(); } catch { /* already stopped */ }
    }
    this.sources = [];
    this.state = "paused";
  }

  destroy(): void {
    for (const s of this.sources) {
      try { s.stop(); } catch { /* ignore */ }
    }
    this.sources = [];
    this.tickCallbacks.clear();
    this.state = "idle";
  }
}
