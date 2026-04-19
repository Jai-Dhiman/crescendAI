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

// Worker message handler — only registers when loaded as a Web Worker (Window is undefined)
if (typeof Window === "undefined") {
  // biome-ignore lint/suspicious/noExplicitAny: dynamic Verovio module
  const toolkitCache = new Map<string, any>();
  // biome-ignore lint/suspicious/noExplicitAny: dynamic WASM module
  let verovioModule: any = null;

  const ready = (async () => {
    const createModule = (
      (await import("verovio/wasm")) as { default: () => Promise<unknown> }
    ).default;
    verovioModule = await createModule();
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
        const { VerovioToolkit } = (await import("verovio/esm")) as {
          VerovioToolkit: new (mod: unknown) => VerovioTk;
        };
        const tk = new VerovioToolkit(verovioModule);
        tk.setOptions({
          pageWidth: 1800,
          adjustPageHeight: true,
          breaks: "none",
          footer: "none",
          header: "none",
        });
        tk.loadZipDataBuffer(msg.bytes);
        toolkitCache.set(msg.pieceId, tk);
      }

      const tk = toolkitCache.get(msg.pieceId)!;
      const svg =
        msg.type === "render_clip"
          ? renderClipSvg(tk, msg.startBar, msg.endBar)
          : renderFullSvg(tk);

      (self as unknown as Worker).postMessage({ requestId: msg.requestId, svg });
    } catch (err) {
      (self as unknown as Worker).postMessage({
        requestId: msg.requestId,
        error: String(err),
      });
    }
  };
}
