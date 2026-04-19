// apps/web/src/lib/score-renderer.ts
import { api } from "./api";

class ScoreRenderer {
  private worker: Worker | null = null;
  private pendingRequests = new Map<
    string,
    { resolve: (svg: string) => void; reject: (err: Error) => void }
  >();
  // bytesCache entries are never evicted by design: sentPieceIds correctness
  // depends on bytesCache remaining populated after a successful fetch.
  private bytesCache = new Map<string, ArrayBuffer>();
  private sentPieceIds = new Set<string>();
  private requestCounter = 0;
  private pendingFetches = new Map<string, Promise<void>>();

  private ensureWorker(): Worker {
    if (!this.worker) {
      if (typeof Worker === "undefined") {
        throw new Error("Web Workers are not available in this environment");
      }
      this.worker = new Worker(new URL("./score-worker.ts", import.meta.url), {
        type: "module",
      });
      this.worker.onmessage = (
        e: MessageEvent<{ requestId: string; svg?: string; error?: string }>,
      ) => {
        const { requestId, svg, error } = e.data;
        const pending = this.pendingRequests.get(requestId);
        if (!pending) return;
        this.pendingRequests.delete(requestId);
        if (error !== undefined) {
          pending.reject(new Error(error));
        } else if (svg !== undefined) {
          pending.resolve(svg);
        } else {
          pending.reject(new Error("Worker returned no svg and no error"));
        }
      };
      this.worker.onerror = (e: ErrorEvent) => {
        const err = new Error(`Score worker crashed: ${e.message}`);
        for (const { reject } of this.pendingRequests.values()) {
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

  async getFull(pieceId: string): Promise<string> {
    await this.ensureBytes(pieceId);
    const worker = this.ensureWorker();
    return new Promise((resolve, reject) => {
      const requestId = `req-${++this.requestCounter}`;
      this.pendingRequests.set(requestId, { resolve, reject });
      const needsBytes = !this.sentPieceIds.has(pieceId);
      const bytes = needsBytes ? this.bytesCache.get(pieceId) : undefined;
      if (needsBytes && bytes === undefined) {
        this.pendingRequests.delete(requestId);
        reject(new Error(`Score bytes missing after fetch for pieceId: ${pieceId}`));
        return;
      }
      if (needsBytes) this.sentPieceIds.add(pieceId);
      worker.postMessage({ type: "render_full", requestId, pieceId, bytes });
    });
  }

  async getClip(pieceId: string, startBar: number, endBar: number): Promise<string> {
    await this.ensureBytes(pieceId);
    const worker = this.ensureWorker();
    return new Promise((resolve, reject) => {
      const requestId = `req-${++this.requestCounter}`;
      this.pendingRequests.set(requestId, { resolve, reject });
      const needsBytes = !this.sentPieceIds.has(pieceId);
      const bytes = needsBytes ? this.bytesCache.get(pieceId) : undefined;
      if (needsBytes && bytes === undefined) {
        this.pendingRequests.delete(requestId);
        reject(new Error(`Score bytes missing after fetch for pieceId: ${pieceId}`));
        return;
      }
      if (!this.sentPieceIds.has(pieceId)) this.sentPieceIds.add(pieceId);
      worker.postMessage({
        type: "render_clip",
        requestId,
        pieceId,
        startBar,
        endBar,
        bytes,
      });
    });
  }
}

export const scoreRenderer = new ScoreRenderer();
