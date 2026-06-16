// apps/web/src/lib/score-renderer.ts
import type { ScoreIR } from "./score-ir";
import { api } from "./api";
import { Sentry } from "./sentry";

type PendingRequest = {
  resolve: (value: unknown) => void;
  reject: (err: Error) => void;
  pieceId: string;
  sentKey?: string;
};

export class ScoreRenderer {
  private worker: Worker | null = null;
  private pendingRequests = new Map<string, PendingRequest>();
  private bytesCache = new Map<string, ArrayBuffer>();
  private sentPieceIds = new Set<string>();
  private requestCounter = 0;
  private pendingFetches = new Map<string, Promise<void>>();
  // Main-thread IR cache populated from load() resolved payload.
  private irCache = new Map<string, ScoreIR>();

  private ensureWorker(): Worker {
    if (!this.worker) {
      if (typeof Worker === "undefined") {
        throw new Error("Web Workers are not available in this environment");
      }
      this.worker = new Worker(new URL("./score-worker.ts", import.meta.url), {
        type: "module",
      });
      this.worker.onmessage = (
        e: MessageEvent<{ requestId: string; payload?: unknown; error?: string }>,
      ) => {
        const { requestId, payload, error } = e.data;
        const pending = this.pendingRequests.get(requestId);
        if (!pending) return;
        this.pendingRequests.delete(requestId);
        if (error !== undefined) {
          if (pending.sentKey !== undefined) this.sentPieceIds.delete(pending.sentKey);
          pending.reject(new Error(error));
        } else {
          pending.resolve(payload);
        }
      };
      this.worker.onerror = (e: ErrorEvent) => {
        const err = new Error(`Score worker crashed: ${e.message}`);
        for (const { reject, sentKey } of this.pendingRequests.values()) {
          if (sentKey !== undefined) this.sentPieceIds.delete(sentKey);
          reject(err);
        }
        this.pendingRequests.clear();
        this.worker = null;
      };
    }
    return this.worker;
  }

  private async ensureBytes(pieceId: string): Promise<void> {
    // sentPieceIds holds composite keys ("pieceId:transpose"), not bare pieceIds,
    // so only bytesCache indicates whether the fetch has already completed.
    if (this.bytesCache.has(pieceId)) return;
    const inflight = this.pendingFetches.get(pieceId);
    if (inflight) return inflight;
    const fetchPromise = (async () => {
      const bytes = await api.scores.getData(pieceId);
      this.bytesCache.set(pieceId, bytes);
    })();
    this.pendingFetches.set(pieceId, fetchPromise);
    try {
      await fetchPromise;
    } finally {
      this.pendingFetches.delete(pieceId);
    }
  }

  private sendRequest<T>(
    pieceId: string,
    msg: Record<string, unknown>,
    bytes?: ArrayBuffer,
    sentKey?: string,
  ): Promise<T> {
    const worker = this.ensureWorker();
    return new Promise((resolve, reject) => {
      const requestId = `req-${++this.requestCounter}`;
      this.pendingRequests.set(requestId, {
        resolve: resolve as (v: unknown) => void,
        reject,
        pieceId,
        sentKey,
      });
      worker.postMessage({ ...msg, requestId, pieceId, bytes });
    });
  }

  // bytes are transpose-independent; bytesCache and pendingFetches key by bare pieceId.
  async load(
    pieceId: string,
    transpose?: number,
  ): Promise<{ ir: ScoreIR; pageSvgs: string[] } | "failed"> {
    const key = `${pieceId}:${transpose ?? 0}`;
    await this.ensureBytes(pieceId);
    const needsBytes = !this.sentPieceIds.has(key);
    const bytes = needsBytes ? this.bytesCache.get(pieceId) : undefined;
    if (needsBytes && bytes === undefined) {
      throw new Error(`Score bytes missing after fetch for pieceId: ${pieceId}`);
    }
    if (needsBytes) this.sentPieceIds.add(key);

    try {
      const payload = await this.sendRequest<{ ir: ScoreIR; pageSvgs: string[] }>(
        pieceId,
        { type: "load", transpose },
        bytes,
        key,
      );
      this.irCache.set(key, payload.ir);
      return payload;
    } catch (err) {
      this.sentPieceIds.delete(key);
      Sentry.captureException(err);
      return "failed";
    }
  }

  getIR(pieceId: string, transpose?: number): ScoreIR | null {
    return this.irCache.get(`${pieceId}:${transpose ?? 0}`) ?? null;
  }

  async getPage(pieceId: string, pageN: number, pageWidth?: number): Promise<string> {
    const worker = this.ensureWorker();
    return new Promise((resolve, reject) => {
      const requestId = `req-${++this.requestCounter}`;
      this.pendingRequests.set(requestId, {
        resolve: resolve as (v: unknown) => void,
        reject,
        pieceId,
      });
      worker.postMessage({ type: "get_page", requestId, pieceId, pageN, pageWidth });
    });
  }

  async getClip(
    pieceId: string,
    startBar: number,
    endBar: number,
  ): Promise<string> {
    const worker = this.ensureWorker();
    return new Promise((resolve, reject) => {
      const requestId = `req-${++this.requestCounter}`;
      this.pendingRequests.set(requestId, {
        resolve: resolve as (v: unknown) => void,
        reject,
        pieceId,
      });
      worker.postMessage({ type: "get_clip", requestId, pieceId, startBar, endBar });
    });
  }

  async getClipPlayback(
    pieceId: string,
    startBar: number,
    endBar: number,
  ): Promise<{ svg: string; ir: ScoreIR; notes: import("./score-worker").ClipNote[] }> {
    const worker = this.ensureWorker();
    return new Promise((resolve, reject) => {
      const requestId = `req-${++this.requestCounter}`;
      this.pendingRequests.set(requestId, {
        resolve: resolve as (v: unknown) => void,
        reject,
        pieceId,
      });
      worker.postMessage({ type: "get_clip_playback", requestId, pieceId, startBar, endBar });
    });
  }
}

export const scoreRenderer = new ScoreRenderer();
