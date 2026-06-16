// apps/web/src/lib/score-worker.dispatch.integration.test.ts
//
// Regression test for the composite-cache-key bug (review P0, #46).
//
// The bug: the worker stored every loaded toolkit under a transpose-suffixed
// key (`${pieceId}:${transpose ?? 0}`) for `load` messages, but looked it up by
// the BARE `pieceId` for every non-load message (get_clip/get_page/...). That is
// a cache MISS for EVERY piece — even untransposed ones cached at `pieceId:0` —
// so every subsequent get_clip returned the "bytes required on first request"
// error and production clip/page rendering was broken.
//
// Why the existing tests missed it:
//   - score-worker.transpose.integration.test.ts drives a FakeWorker that echoes
//     a canned payload — it never exercises the real onmessage/toolkitCache.
//   - The real-Verovio tests call loadPiece()/processRenderClipRequest() DIRECTLY,
//     bypassing message dispatch and the toolkitCache entirely.
//
// This test drives the REAL worker message dispatch + toolkitCache: it sends a
// `load` message, then a `get_clip`/`get_page` message for the SAME piece, and
// asserts a real SVG payload comes back (NOT the error). It would fail against
// the pre-fix bare-pieceId lookup.
import { readFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { afterEach, describe, expect, it, vi } from "vitest";

const FIXTURE_PATH = resolve(
  dirname(fileURLToPath(import.meta.url)),
  "../../public/scores/czerny-op299-no1.mxl",
);

function freshBuf(): ArrayBuffer {
  const bytes = readFileSync(FIXTURE_PATH);
  const buf = new ArrayBuffer(bytes.byteLength);
  new Uint8Array(buf).set(bytes);
  return buf;
}

type WorkerOut = { requestId: string; payload?: unknown; error?: string };

// The worker reads `self` and `typeof window` as runtime globals. To drive its
// real onmessage handler we must SIMULATE the Web Worker global scope for the
// whole test body: `window` removed (so the handler registers + so verovio's
// dynamic imports take the worker path) and `self` pointing at a fake that
// captures the handler and collects postMessage outputs. We restore the real
// globals in afterEach so the rest of the suite is unaffected.
let restoreGlobals: (() => void) | null = null;

// Boots the real worker module's onmessage handler against the fake self.
async function bootWorker(): Promise<{
  send: (msg: Record<string, unknown>) => Promise<WorkerOut>;
}> {
  const realWindow = (globalThis as { window?: unknown }).window;
  const realSelf = (globalThis as { self?: unknown }).self;

  type Handler = (e: MessageEvent) => void | Promise<void>;
  let handler: Handler | null = null;
  const outbox: WorkerOut[] = [];
  const fakeSelf = {
    set onmessage(fn: Handler) {
      handler = fn;
    },
    postMessage(msg: WorkerOut) {
      outbox.push(msg);
    },
  };

  // Simulate the Web Worker global scope: no window, self present.
  delete (globalThis as { window?: unknown }).window;
  (globalThis as { self?: unknown }).self = fakeSelf;
  restoreGlobals = () => {
    if (realWindow !== undefined) {
      (globalThis as { window?: unknown }).window = realWindow;
    }
    (globalThis as { self?: unknown }).self = realSelf;
  };

  // Fresh module so the top-level `if (typeof window === "undefined")` block runs
  // and registers the handler against our fakeSelf.
  vi.resetModules();
  await import("./score-worker");

  if (handler === null) {
    throw new Error("score-worker did not register an onmessage handler");
  }
  // Non-null const so the nested `send` closure does not re-widen to Handler | null.
  const boundHandler: Handler = handler;

  let counter = 0;
  const send = async (msg: Record<string, unknown>): Promise<WorkerOut> => {
    const requestId = `req-${++counter}`;
    const before = outbox.length;
    await boundHandler({ data: { ...msg, requestId } } as MessageEvent);
    const reply = outbox.slice(before).find((o) => o.requestId === requestId);
    if (!reply) throw new Error(`no reply for ${requestId}`);
    return reply;
  };

  return { send };
}

describe("score-worker real message dispatch + toolkitCache (P0 regression)", () => {
  afterEach(() => {
    restoreGlobals?.();
    restoreGlobals = null;
    vi.resetModules();
  });

  it("(a) load then get_clip with no transpose returns a clip SVG (not 'bytes required')", async () => {
    const { send } = await bootWorker();

    const loadReply = await send({
      type: "load",
      pieceId: "dispatch-notranspose",
      bytes: freshBuf(),
    });
    expect(loadReply.error).toBeUndefined();
    expect(loadReply.payload).toBeDefined();

    // get_clip for the SAME piece, no transpose. Pre-fix this looked up the bare
    // pieceId and missed the `dispatch-notranspose:0` toolkit entry → error.
    const clipReply = await send({
      type: "get_clip",
      pieceId: "dispatch-notranspose",
      startBar: 1,
      endBar: 2,
    });
    expect(clipReply.error).toBeUndefined();
    expect(typeof clipReply.payload).toBe("string");
    expect(clipReply.payload as string).toContain("<svg");

    // get_page for the SAME piece, no transpose — same cache-key path.
    const pageReply = await send({
      type: "get_page",
      pieceId: "dispatch-notranspose",
      pageN: 1,
    });
    expect(pageReply.error).toBeUndefined();
    expect(typeof pageReply.payload).toBe("string");
    expect(pageReply.payload as string).toContain("<svg");
  }, 60_000);

  it("(b) load then get_clip with transpose:N resolves to the transposed toolkit", async () => {
    const { send } = await bootWorker();

    const loadReply = await send({
      type: "load",
      pieceId: "dispatch-transpose",
      bytes: freshBuf(),
      transpose: 2,
    });
    expect(loadReply.error).toBeUndefined();
    expect(loadReply.payload).toBeDefined();

    // get_clip must carry the same transpose so the composite key matches
    // `dispatch-transpose:2`.
    const clipReply = await send({
      type: "get_clip",
      pieceId: "dispatch-transpose",
      startBar: 1,
      endBar: 2,
      transpose: 2,
    });
    expect(clipReply.error).toBeUndefined();
    expect(typeof clipReply.payload).toBe("string");
    expect(clipReply.payload as string).toContain("<svg");

    // A get_clip WITHOUT transpose keys `dispatch-transpose:0`, which was never
    // loaded — so it must correctly report "bytes required". This proves the key
    // is genuinely transpose-scoped (not a bare-pieceId match against the :2 entry).
    const missReply = await send({
      type: "get_clip",
      pieceId: "dispatch-transpose",
      startBar: 1,
      endBar: 2,
    });
    expect(missReply.payload).toBeUndefined();
    expect(missReply.error).toContain("bytes required");
  }, 60_000);
});
