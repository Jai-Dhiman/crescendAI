// apps/web/src/lib/score-worker.transpose.integration.test.ts
// Real-Verovio integration: loadPiece's optional transpose param must change
// the engraving, and transpose:0 must be byte-identical to omitting it.
import { readFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { describe, expect, it } from "vitest";

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
      //    Pattern: " id-" followed by lowercase alphanumeric.
      out = out.replace(/ id-[a-z0-9]+/g, "");
      return out;
    };
    expect(stripIds(zero.pageSvgs[0])).toBe(stripIds(omitted.pageSvgs[0]));
  }, 60_000);
});
