// ScoreHost JS facade — exposes window.ScoreHost for WKWebView consumption.
// Fetches .mxl files from the bundled public/scores/ directory (not the live API)
// and delegates engraving to the score-worker via the standard message protocol.

import type { InlineComponent, ScoreHighlightConfig } from "../lib/types";
import type { ScoreIR } from "../lib/score-ir";

interface WorkerReply {
  requestId: string;
  payload?: unknown;
  error?: string;
}

interface LoadPayload {
  ir: ScoreIR;
  pageSvgs: string[];
}

let worker: Worker | null = null;
let requestCounter = 0;
const pending = new Map<string, { resolve: (v: unknown) => void; reject: (e: Error) => void }>();

// Per-pieceId cache of the load result so showArtifact can call renderScoreHighlight
// without re-loading every time.
const loadCache = new Map<string, LoadPayload>();
const loadPromises = new Map<string, Promise<void>>();

function getWorker(): Worker {
  if (worker) return worker;
  worker = new Worker(new URL("../lib/score-worker.ts", import.meta.url), {
    type: "module",
  });
  worker.onmessage = (e: MessageEvent<WorkerReply>) => {
    const { requestId, payload, error } = e.data;
    const entry = pending.get(requestId);
    if (!entry) return;
    pending.delete(requestId);
    if (error !== undefined) {
      entry.reject(new Error(error));
    } else {
      entry.resolve(payload);
    }
  };
  worker.onerror = (e: ErrorEvent) => {
    const err = new Error(`ScoreHost worker crashed: ${e.message}`);
    for (const entry of pending.values()) {
      entry.reject(err);
    }
    pending.clear();
    worker = null;
  };
  return worker;
}

function sendRequest<T>(msg: Record<string, unknown>, bytes?: ArrayBuffer): Promise<T> {
  const w = getWorker();
  return new Promise<T>((resolve, reject) => {
    const requestId = `req-${++requestCounter}`;
    pending.set(requestId, {
      resolve: resolve as (v: unknown) => void,
      reject,
    });
    if (bytes !== undefined) {
      w.postMessage({ ...msg, requestId, bytes }, [bytes]);
    } else {
      w.postMessage({ ...msg, requestId });
    }
  });
}

// Emit events to the WKWebView message handler when present.
function emit(type: string, payload?: Record<string, unknown>): void {
  const wk = (window as unknown as {
    webkit?: {
      messageHandlers?: {
        scoreHostEvents?: { postMessage: (msg: unknown) => void };
      };
    };
  }).webkit?.messageHandlers?.scoreHostEvents;
  if (wk) {
    wk.postMessage({ type, payload: payload ?? {} });
  }
}

function getContainer(): HTMLElement {
  const el = document.getElementById("scorehost-container");
  if (!el) {
    throw new Error("scorehost-container element not found in DOM");
  }
  return el;
}

async function fetchScoreBytes(pieceId: string): Promise<ArrayBuffer> {
  const url = `./scores/${pieceId}.mxl`;
  const res = await fetch(url);
  if (!res.ok) {
    throw new Error(`Failed to fetch score for ${pieceId}: HTTP ${res.status}`);
  }
  return res.arrayBuffer();
}

async function ensureLoaded(pieceId: string): Promise<void> {
  if (loadCache.has(pieceId)) return;
  const inflight = loadPromises.get(pieceId);
  if (inflight) {
    await inflight;
    return;
  }
  const promise = (async () => {
    const bytes = await fetchScoreBytes(pieceId);
    const payload = await sendRequest<LoadPayload>({ type: "load", pieceId }, bytes);
    loadCache.set(pieceId, payload);
  })();
  loadPromises.set(pieceId, promise);
  try {
    await promise;
  } finally {
    loadPromises.delete(pieceId);
  }
}

async function renderScoreHighlight(config: ScoreHighlightConfig): Promise<void> {
  const { pieceId } = config;

  await ensureLoaded(pieceId);

  const container = getContainer();
  container.innerHTML = "";

  let svgHtml: string;
  if (config.highlights.length > 0) {
    const [startBar, endBar] = config.highlights[0].bars;
    svgHtml = await sendRequest<string>({ type: "get_clip", pieceId, startBar, endBar });
  } else {
    const cached = loadCache.get(pieceId);
    if (!cached) {
      throw new Error(`renderScoreHighlight: no cached load for pieceId: ${pieceId}`);
    }
    svgHtml = cached.pageSvgs[0] ?? "";
  }

  const wrapper = document.createElement("div");
  wrapper.style.cssText = "position:relative;width:100%;";
  wrapper.innerHTML = svgHtml;
  container.appendChild(wrapper);

  const noteCount = wrapper.querySelectorAll("use").length;
  emit("rendered", { noteCount });
}

const ScoreHostImpl = {
  async ready(): Promise<void> {
    emit("ready");
  },

  async load(pieceId: string): Promise<{ ok: true }> {
    try {
      await ensureLoaded(pieceId);
      return { ok: true };
    } catch (err) {
      const reason = err instanceof Error ? err.message : String(err);
      emit("error", { reason });
      throw err;
    }
  },

  async showArtifact(json: string): Promise<void> {
    let component: InlineComponent;
    try {
      component = JSON.parse(json) as InlineComponent;
    } catch (err) {
      throw new Error(
        `showArtifact: invalid JSON — ${err instanceof Error ? err.message : String(err)}`,
      );
    }

    try {
      if (component.type === "score_highlight") {
        await renderScoreHighlight(component.config);
      } else if (component.type === "play_passage") {
        throw new Error("play_passage not yet implemented");
      } else if (component.type === "exercise_set") {
        throw new Error("exercise_set scoreClip not yet implemented");
      } else {
        throw new Error(
          `showArtifact: unsupported artifact type: ${(component as { type: string }).type}`,
        );
      }
    } catch (err) {
      const reason = err instanceof Error ? err.message : String(err);
      emit("error", { reason });
      throw err;
    }
  },

  async play(): Promise<void> {
    throw new Error("play not yet implemented");
  },

  async stop(): Promise<void> {
    emit("playback", { state: "stopped" });
  },

  async setTempo(_factor: number): Promise<void> {
    throw new Error("setTempo not yet implemented");
  },
};

(window as unknown as { ScoreHost: typeof ScoreHostImpl }).ScoreHost = ScoreHostImpl;
