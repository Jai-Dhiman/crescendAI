// apps/web/src/lib/score-renderer.ts
import type { ScoreIR } from "./score-ir";
import { api } from "./api";
import { Sentry } from "./sentry";

type PendingRequest = {
  resolve: (value: unknown) => void;
  reject: (err: Error) => void;
  pieceId: string;
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
          this.sentPieceIds.delete(pending.pieceId);
          pending.reject(new Error(error));
        } else {
          pending.resolve(payload);
        }
      };
      this.worker.onerror = (e: ErrorEvent) => {
        const err = new Error(`Score worker crashed: ${e.message}`);
        for (const { reject, pieceId } of this.pendingRequests.values()) {
          this.sentPieceIds.delete(pieceId);
          reject(err);
        }
        this.pendingRequests.clear();
        this.worker = null;
      };
    }
    return this.worker;
  }

  private async ensureBytes(pieceId: string): Promise<void> {
    if (this.sentPieceIds.has(pieceId) || this.bytesCache.has(pieceId)) return;
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
  ): Promise<T> {
    const worker = this.ensureWorker();
    return new Promise((resolve, reject) => {
      const requestId = `req-${++this.requestCounter}`;
      this.pendingRequests.set(requestId, {
        resolve: resolve as (v: unknown) => void,
        reject,
        pieceId,
      });
      worker.postMessage({ ...msg, requestId, pieceId, bytes });
    });
  }

  async load(
    pieceId: string,
  ): Promise<{ ir: ScoreIR; pageSvgs: string[] } | "failed"> {
    await this.ensureBytes(pieceId);
    const needsBytes = !this.sentPieceIds.has(pieceId);
    const bytes = needsBytes ? this.bytesCache.get(pieceId) : undefined;
    if (needsBytes && bytes === undefined) {
      throw new Error(`Score bytes missing after fetch for pieceId: ${pieceId}`);
    }
    if (needsBytes) this.sentPieceIds.add(pieceId);

    try {
      const payload = await this.sendRequest<{ ir: ScoreIR; pageSvgs: string[] }>(
        pieceId,
        { type: "load" },
        bytes,
      );
      this.irCache.set(pieceId, payload.ir);
      return payload;
    } catch (err) {
      Sentry.captureException(err);
      return "failed";
    }
  }

  getIR(pieceId: string): ScoreIR | null {
    return this.irCache.get(pieceId) ?? null;
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
}

export const scoreRenderer = new ScoreRenderer();
