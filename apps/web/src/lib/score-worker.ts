// apps/web/src/lib/score-worker.ts

// biome-ignore lint/suspicious/noExplicitAny: Verovio has no exported TS types
type VerovioTk = any;

export function renderClipSvg(tk: VerovioTk, startBar: number, endBar: number): string {
  tk.select({ measureRange: `${startBar}-${endBar}` });
  return tk.renderToSVG(1) as string;
}

export function renderFullSvg(tk: VerovioTk): string {
  tk.select({});
  return tk.renderToSVG(1) as string;
}

type WorkerInMsg =
  | {
      type: "render_clip";
      requestId: string;
      pieceId: string;
      startBar: number;
      endBar: number;
      bytes?: ArrayBuffer;
    }
  | { type: "render_full"; requestId: string; pieceId: string; bytes?: ArrayBuffer };

// Worker message handler — only registers when loaded as a Web Worker (window is undefined)
if (typeof window === "undefined") {
  // biome-ignore lint/suspicious/noExplicitAny: dynamic Verovio module
  const toolkitCache = new Map<string, any>();
  // biome-ignore lint/suspicious/noExplicitAny: dynamic WASM module
  let verovioModule: any = null;
  // biome-ignore lint/suspicious/noExplicitAny: dynamic Verovio ESM class
  let VerovioToolkitClass: new (mod: unknown) => VerovioTk = null as any;

  const ready = (async () => {
    const [wasm, esm] = await Promise.all([
      import("verovio/wasm") as Promise<{ default: () => Promise<unknown> }>,
      import("verovio/esm") as Promise<{ VerovioToolkit: new (mod: unknown) => VerovioTk }>,
    ]);
    verovioModule = await wasm.default();
    VerovioToolkitClass = esm.VerovioToolkit;
  })();

  (self as unknown as Worker).onmessage = async (event: MessageEvent<WorkerInMsg>) => {
    const msg = event.data;
    try {
      await ready;

      if (!toolkitCache.has(msg.pieceId)) {
        if (!msg.bytes) {
          (self as unknown as Worker).postMessage({
            requestId: msg.requestId,
            error: "bytes required on first request",
          });
          return;
        }
        const tk = new VerovioToolkitClass(verovioModule);
        tk.setOptions({
          pageWidth: 1800,
          adjustPageHeight: true,
          breaks: "none",
          footer: "none",
          header: "none",
        });
        const loaded = tk.loadZipDataBuffer(msg.bytes);
        if (!loaded) {
          (self as unknown as Worker).postMessage({
            requestId: msg.requestId,
            error: "Verovio could not parse MXL data — file may be corrupt or not a valid MusicXML ZIP",
          });
          return;
        }
        toolkitCache.set(msg.pieceId, tk);
      }

      const tk = toolkitCache.get(msg.pieceId)!;
      const svg =
        msg.type === "render_clip"
          ? renderClipSvg(tk, msg.startBar, msg.endBar)
          : renderFullSvg(tk);

      (self as unknown as Worker).postMessage({ requestId: msg.requestId, svg });
    } catch (err) {
      const errorMsg =
        typeof WebAssembly !== "undefined" && err instanceof WebAssembly.Exception
          ? `Verovio WASM exception (${msg.type} for ${msg.pieceId}) — MXL data may be corrupt or incompatible`
          : String(err);
      (self as unknown as Worker).postMessage({
        requestId: msg.requestId,
        error: errorMsg,
      });
    }
  };
}
