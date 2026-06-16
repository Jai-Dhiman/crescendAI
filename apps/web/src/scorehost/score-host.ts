// ScoreHost JS facade — exposes window.ScoreHost for WKWebView consumption.
// Fetches .mxl files from the bundled public/scores/ directory or from the live
// API when window.__SCOREHOST_API_BASE is set (used from file:// / custom schemes).
// Delegates engraving to the score-worker via the standard message protocol.

import type { InlineComponent, PlayPassageConfig, ScoreHighlightConfig } from "../lib/types";
import type { ScoreIR } from "../lib/score-ir";
import { LoopPlayer } from "../lib/loop-player";
import { ScoreCursor } from "../lib/score-cursor";
import type { ClipPlaybackResult } from "../lib/score-worker";

// When scorehost is loaded from file:// or a custom scheme (e.g. iOS WKWebView),
// relative /api/ URLs fail. Callers may set window.__SCOREHOST_API_BASE before
// any ScoreHost calls to prepend an absolute origin for those requests.
// The base URL is read lazily on each fetch call so it can be set after module load.
(function patchFetchForScorehost(): void {
  const origFetch = window.fetch.bind(window);
  window.fetch = function(input: RequestInfo | URL, init?: RequestInit) {
    const url =
      typeof input === "string"
        ? input
        : input instanceof URL
          ? input.toString()
          : input.url;
    const apiBase = (window as unknown as { __SCOREHOST_API_BASE?: string }).__SCOREHOST_API_BASE;
    if (apiBase && url.startsWith("/api/")) {
      return origFetch(`${apiBase}${url}`, init);
    }
    return origFetch(input, init);
  };
}());

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

// Module-level state for the active loop player and cursor.
let activePlayer: LoopPlayer | null = null;
let activeCursor: ScoreCursor | null = null;
let currentPieceId: string | null = null;

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
  const apiBase = (window as unknown as { __SCOREHOST_API_BASE?: string }).__SCOREHOST_API_BASE;
  const url = apiBase
    ? `${apiBase}/api/scores/${pieceId}/data`
    : `./scores/${pieceId}.mxl`;
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

async function renderPlayPassage(config: PlayPassageConfig): Promise<void> {
  const pieceId = config.pieceId;
  await ensureLoaded(pieceId);

  const [startBar, endBar] = config.bars;
  const clipResult = await sendRequest<ClipPlaybackResult | "failed">(
    { type: "get_clip_playback", pieceId, startBar, endBar },
  );
  if (clipResult === "failed") {
    throw new Error(`renderPlayPassage: get_clip_playback failed for ${pieceId} bars ${startBar}-${endBar}`);
  }

  // Tear down any previous player/cursor before mutating DOM.
  if (activePlayer) {
    activePlayer.stop();
    activePlayer = null;
  }
  if (activeCursor) {
    activeCursor.stop();
    activeCursor = null;
  }

  currentPieceId = pieceId;

  const container = getContainer();
  container.innerHTML = "";

  // Render clip SVG.
  const svgWrapper = document.createElement("div");
  svgWrapper.style.cssText = "position:relative;width:100%;";
  svgWrapper.innerHTML = clipResult.svg;
  container.appendChild(svgWrapper);

  const noteCount = svgWrapper.querySelectorAll("use").length;

  // Build transport controls inside the WebView so AudioContext user-gesture
  // chain is preserved when the user taps the Play button.
  const transport = document.createElement("div");
  transport.id = "loop-transport";
  transport.style.cssText = "display:flex;align-items:center;gap:8px;padding:8px 0;";

  const playBtn = document.createElement("button");
  playBtn.id = "loop-play-btn";
  playBtn.textContent = "Play";
  playBtn.type = "button";
  playBtn.style.cssText = "padding:6px 14px;border-radius:6px;border:1px solid #ccc;cursor:pointer;";

  const tempoLabel = document.createElement("span");
  tempoLabel.textContent = "Tempo:";
  tempoLabel.style.cssText = "font-size:12px;";

  const tempoSlider = document.createElement("input");
  tempoSlider.type = "range";
  tempoSlider.id = "loop-tempo-slider";
  tempoSlider.min = "0.5";
  tempoSlider.max = "1.5";
  tempoSlider.step = "0.05";
  tempoSlider.value = "1.0";

  transport.appendChild(playBtn);
  transport.appendChild(tempoLabel);
  transport.appendChild(tempoSlider);
  container.appendChild(transport);

  const ir = clipResult.ir;
  const notes = clipResult.notes;
  const beatsPerBar = 4; // default 4/4; BarIR does not carry time signature yet
  console.warn(JSON.stringify({
    msg: "beatsPerBar defaulting to 4 — BarIR does not carry time signature; tempo will be wrong for non-4/4 pieces",
    pieceId,
    bars: config.bars,
  }));
  const bpmAtUnity = 120;

  let isPlaying = false;

  playBtn.addEventListener("click", async () => {
    if (isPlaying) {
      activePlayer?.stop();
      activeCursor?.stop();
      isPlaying = false;
      playBtn.textContent = "Play";
      emit("playback", { state: "stopped" });
      return;
    }

    const ctx = new AudioContext();
    const SOUNDFONT_PATH = "./soundfonts/acoustic_grand_piano-mp3.js";

    if (activePlayer) {
      activePlayer.stop();
    }
    activePlayer = new LoopPlayer({
      ctx,
      instrumentUrl: SOUNDFONT_PATH,
      clipIR: ir,
      clipNotes: notes,
      beatsPerBar,
      bpmAtUnity,
      tempoFactor: parseFloat(tempoSlider.value),
    });

    activeCursor = new ScoreCursor({
      pieceId,
      container: svgWrapper,
      ir,
      qstampSource: () => activePlayer?.qstampSource() ?? null,
    });

    await activePlayer.play();
    activeCursor.start();
    isPlaying = true;
    playBtn.textContent = "Stop";
    emit("playback", { state: "playing" });
  });

  tempoSlider.addEventListener("input", () => {
    const factor = parseFloat(tempoSlider.value);
    activePlayer?.setTempoFactor(factor);
  });

  emit("rendered", { noteCount });
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
        await renderPlayPassage(component.config);
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
    const playBtn = document.getElementById("loop-play-btn") as HTMLButtonElement | null;
    if (!playBtn) {
      throw new Error("play: no active loop transport — call showArtifact(play_passage) first");
    }
    playBtn.click();
  },

  async stop(): Promise<void> {
    if (activePlayer) {
      activePlayer.stop();
      activePlayer = null;
    }
    if (activeCursor) {
      activeCursor.stop();
      activeCursor = null;
    }
    emit("playback", { state: "stopped" });
  },

  async setTempo(factor: number): Promise<void> {
    const slider = document.getElementById("loop-tempo-slider") as HTMLInputElement | null;
    if (slider) {
      slider.value = String(factor);
      slider.dispatchEvent(new Event("input"));
    }
    activePlayer?.setTempoFactor(factor);
  },
};

(window as unknown as { ScoreHost: typeof ScoreHostImpl }).ScoreHost = ScoreHostImpl;
