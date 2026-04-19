// apps/web/src/lib/score-renderer.ts
import { api } from "./api";

class ScoreRenderer {
  private worker: Worker | null = null;
  private pendingRequests = new Map<
    string,
    { resolve: (svg: string) => void; reject: (err: Error) => void }
  >();
  private bytesCache = new Map<string, ArrayBuffer>();
  private sentPieceIds = new Set<string>();
  private requestCounter = 0;

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
        } else {
          pending.resolve(svg ?? "");
        }
      };
    }
    return this.worker;
  }

  private async ensureBytes(pieceId: string): Promise<void> {
    if (this.sentPieceIds.has(pieceId) || this.bytesCache.has(pieceId)) return;
    const bytes = await api.scores.getData(pieceId);
    this.bytesCache.set(pieceId, bytes);
  }

  async getClip(pieceId: string, startBar: number, endBar: number): Promise<string> {
    await this.ensureBytes(pieceId);
    const worker = this.ensureWorker();
    return new Promise((resolve, reject) => {
      const requestId = `req-${++this.requestCounter}`;
      this.pendingRequests.set(requestId, { resolve, reject });
      const bytes = !this.sentPieceIds.has(pieceId)
        ? this.bytesCache.get(pieceId)
        : undefined;
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
