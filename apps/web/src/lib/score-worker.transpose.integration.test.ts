// apps/web/src/lib/score-worker.transpose.integration.test.ts
// Real-Verovio integration: loadPiece's optional transpose param must change
// the engraving, and transpose:0 must be byte-identical to omitting it.
import { readFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { describe, expect, it } from "vitest";
import { ScoreRenderer } from "./score-renderer";

const FIXTURE_PATH = resolve(
  dirname(fileURLToPath(import.meta.url)),
  "../../public/scores/czerny-op299-no1.mxl",
);

async function makeBindings() {
  // biome-ignore lint/suspicious/noExplicitAny: dynamic ESM
  const esm = (await import("verovio/esm")) as any;
  // biome-ignore lint/suspicious/noExplicitAny: dynamic ESM
  const wasm = (await import("verovio/wasm")) as any;
  const VerovioToolkit = esm.VerovioToolkit ?? esm.default?.VerovioToolkit;
  const VerovioModule = wasm.default ?? wasm;
  const mod = await VerovioModule();
  return { module: mod, ToolkitClass: VerovioToolkit as any };
}

function freshBuf(): ArrayBuffer {
  const bytes = readFileSync(FIXTURE_PATH);
  const buf = new ArrayBuffer(bytes.byteLength);
  new Uint8Array(buf).set(bytes);
  return buf;
}

describe("loadPiece transpose param", () => {
  it("transpose:2 yields a different page-1 SVG than transpose:0", async () => {
    const bindings = await makeBindings();
    const { loadPiece } = await import("./score-worker");

    const base = await loadPiece(freshBuf(), bindings, "transpose-fixture-0", 0);
    const up = await loadPiece(freshBuf(), bindings, "transpose-fixture-2", 2);

    expect(base).not.toBe("failed");
    expect(up).not.toBe("failed");
    if (base === "failed" || up === "failed") return;

    expect(up.pageSvgs[0]).not.toBe(base.pageSvgs[0]);
  }, 60_000);

  it("transpose:0 is structurally identical to omitting the argument (real-piece lock)", async () => {
    const bindings = await makeBindings();
    const { loadPiece } = await import("./score-worker");

    // Verovio randomizes element ids on every loadData call (see the existing
    // score-worker.integration.test.ts "Cache eviction" test), so raw SVG bytes
    // differ even for identical input. Strip ids before comparing so this lock
    // verifies the ENGRAVING is identical between transpose:0 and no-transpose --
    // i.e. transpose:0 is a true no-op for real pieces.
    const omitted = await loadPiece(freshBuf(), bindings, "lock-omit");
    const zero = await loadPiece(freshBuf(), bindings, "lock-zero", 0);

    expect(omitted).not.toBe("failed");
    expect(zero).not.toBe("failed");
    if (omitted === "failed" || zero === "failed") return;

    // Verovio randomizes ids on every render. They appear in four forms:
    //   1. id="<randomId>" attributes (on SVG elements)
    //   2. xlink:href="#<symbolName>-<rootSuffix>" (defs references)
    //   3. <style> CSS selectors: #<rootSuffix> g.ending { ... }
    //   4. class="tie id-<perElemSuffix> spanning" (per-element random tokens)
    // Strip all four with hardcoded patterns so comparison checks engraving
    // structure only, not the random tokens.
    const stripIds = (svg: string) => {
      // 1. Remove all id="..." attributes
      let out = svg.replace(/\bid="[^"]*"/g, 'id=""');
      // 2. Extract the root suffix from xlink:href="#E050-<suffix>" pattern
      //    and collapse it; use split/join to avoid dynamic RegExp (ReDoS).
      const hrefMatch = out.match(/xlink:href="#[A-Z0-9]+-([a-z0-9]+)"/);
      if (hrefMatch) {
        out = out.split(hrefMatch[1]).join("VSUFFIX");
      }
      // 3. Strip per-element random id tokens from class attributes,
      //    e.g. class="tie id-jiay29h spanning" → class="tie spanning"
      //    Scoped to class="..." values only to avoid stripping " id-<word>"
      //    patterns that appear in text content or other attributes.
      //    Inner /g removes ALL id- fragments per class (Verovio may emit multiple).
      out = out.replace(/class="([^"]*)"/g, (_m, cls) => `class="${cls.replace(/ id-[a-z0-9]+/g, "")}"`)
      return out;
    };
    expect(stripIds(zero.pageSvgs[0])).toBe(stripIds(omitted.pageSvgs[0]));
  }, 60_000);
});

describe("ScoreRenderer.load forwards transpose into the worker message", () => {
  it("posts transpose in the load message", async () => {
    // biome-ignore lint/suspicious/noExplicitAny: test capture array
    const posted: any[] = [];
    class FakeWorker {
      onmessage: ((e: MessageEvent) => void) | null = null;
      onerror: ((e: ErrorEvent) => void) | null = null;
      // biome-ignore lint/suspicious/noExplicitAny: synthetic message
      postMessage(msg: any) {
        posted.push(msg);
        queueMicrotask(() => {
          this.onmessage?.({
            data: {
              requestId: msg.requestId,
              payload: {
                ir: { pieceId: `${msg.pieceId}`, verovioVersion: "", bars: [], pages: [], notes: {}, pageWidth: 2400 },
                pageSvgs: ["<svg/>"],
              },
            },
          } as MessageEvent);
        });
      }
      terminate() {}
    }
    // @ts-expect-error override global Worker for the test
    globalThis.Worker = FakeWorker;

    const { api } = await import("./api");
    const orig = api.scores.getData;
    // @ts-expect-error stub network fetch
    api.scores.getData = async () => new ArrayBuffer(8);

    try {
      const r = new ScoreRenderer();
      await r.load("hanon_001", 2);
      const loadMsg = posted.find((m) => m.type === "load");
      expect(loadMsg).toBeDefined();
      expect(loadMsg.transpose).toBe(2);
    } finally {
      // @ts-expect-error restore
      api.scores.getData = orig;
    }
  });

  it("clears composite sentPieceIds key on worker error so retry re-sends bytes (transpose case)", async () => {
    // biome-ignore lint/suspicious/noExplicitAny: test capture array
    const posted: any[] = [];
    let callCount = 0;
    let workerRef: { onmessage: ((e: MessageEvent) => void) | null } = { onmessage: null };

    class FakeWorker {
      onmessage: ((e: MessageEvent) => void) | null = null;
      onerror: ((e: ErrorEvent) => void) | null = null;
      constructor() {
        // Keep a reference so the test can drive responses manually
        workerRef = this;
      }
      // biome-ignore lint/suspicious/noExplicitAny: synthetic message
      postMessage(msg: any) {
        posted.push(msg);
        callCount++;
        const self = this;
        if (callCount === 1) {
          // First call: reply with an error
          queueMicrotask(() => {
            self.onmessage?.({
              data: { requestId: msg.requestId, error: "boom" },
            } as MessageEvent);
          });
        } else {
          // Second call: reply with success
          queueMicrotask(() => {
            self.onmessage?.({
              data: {
                requestId: msg.requestId,
                payload: {
                  ir: { pieceId: msg.pieceId, verovioVersion: "", bars: [], pages: [], notes: {}, pageWidth: 2400 },
                  pageSvgs: ["<svg/>"],
                },
              },
            } as MessageEvent);
          });
        }
      }
      terminate() {}
    }
    // @ts-expect-error override global Worker for the test
    globalThis.Worker = FakeWorker;

    const { api } = await import("./api");
    const orig = api.scores.getData;
    // @ts-expect-error stub network fetch
    api.scores.getData = async () => new ArrayBuffer(8);

    try {
      const r = new ScoreRenderer();

      // First load: should fail (worker returns error), returns "failed"
      const firstResult = await r.load("hanon_001", 2);
      expect(firstResult).toBe("failed");

      // Second load: sentPieceIds["hanon_001:2"] must have been cleared,
      // so bytes are re-sent (the key indicator that the composite key was cleaned up).
      const secondResult = await r.load("hanon_001", 2);
      expect(secondResult).not.toBe("failed");

      const secondMsg = posted[1];
      expect(secondMsg).toBeDefined();
      // bytes present proves sentPieceIds was properly cleared for the composite key
      expect(secondMsg.bytes).toBeInstanceOf(ArrayBuffer);
    } finally {
      // @ts-expect-error restore
      api.scores.getData = orig;
    }
  });
});
