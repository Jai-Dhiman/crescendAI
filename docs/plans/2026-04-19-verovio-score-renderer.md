# Verovio Score Renderer Implementation Plan

> **For the build agent:** Dispatch each task group in parallel (one subagent per task).
> Do NOT start execution until /challenge returns VERDICT: PROCEED.

**Goal:** Replace the broken OSMD-based score rendering with Verovio WASM in a Web Worker, serving all score consumers from a two-method singleton interface.
**Spec:** docs/specs/2026-04-19-verovio-score-renderer-design.md
**Style:** Follow `apps/web/CLAUDE.md` and `apps/api/TS_STYLE.md`

---

## Task Groups

```
Group A (parallel): Task 1 (score-worker.ts), Task 2 (sandbox resize variants)
Group B (after A):  Task 3 (score-renderer.ts basic getClip)
Group C (after B):  Task 4 (score-renderer.ts fetch deduplication)
Group D (after C):  Task 5 (score-renderer.ts getFull)
Group E (after D):  Task 6 (score-renderer.ts error propagation)
Group F (parallel, after E): Task 7 (ScoreHighlightCard), Task 8 (ScorePanel)
Group G (after F):  Task 9 (cleanup)
```

---

### Task 1: score-worker.ts — Verovio pure render functions + Worker handler

**Group:** A (parallel with Task 2)

**Behavior being verified:** `renderClipSvg` calls `select({ measureRange })` with the correct bar range string and returns the SVG string; `renderFullSvg` clears selection and returns the SVG string.

**Interface under test:** `renderClipSvg(tk, startBar, endBar): string` and `renderFullSvg(tk): string` exported from `score-worker.ts`.

**Files:**
- Create: `apps/web/src/lib/score-worker.ts`
- Create: `apps/web/src/lib/score-worker.test.ts`

---

- [ ] **Step 1: Write the failing test**

```typescript
// apps/web/src/lib/score-worker.test.ts
import { beforeEach, describe, expect, it, vi } from "vitest";

const mockSelect = vi.fn();
const mockRenderToSVG = vi.fn().mockReturnValue("<svg>clip-svg</svg>");
const mockLoadZipDataBuffer = vi.fn();

const fakeTk = {
  select: mockSelect,
  renderToSVG: mockRenderToSVG,
  loadZipDataBuffer: mockLoadZipDataBuffer,
  setOptions: vi.fn(),
};

beforeEach(() => {
  vi.clearAllMocks();
});

describe("renderClipSvg", () => {
  it("selects the requested measure range and returns the SVG string", async () => {
    const { renderClipSvg } = await import("./score-worker");
    // biome-ignore lint/suspicious/noExplicitAny: test mock
    const result = renderClipSvg(fakeTk as any, 3, 6);
    expect(mockSelect).toHaveBeenCalledWith({ measureRange: "3-6" });
    expect(mockRenderToSVG).toHaveBeenCalledWith(1);
    expect(result).toBe("<svg>clip-svg</svg>");
  });
});

describe("renderFullSvg", () => {
  it("clears any previous selection and returns the SVG string", async () => {
    const { renderFullSvg } = await import("./score-worker");
    mockRenderToSVG.mockReturnValue("<svg>full-svg</svg>");
    // biome-ignore lint/suspicious/noExplicitAny: test mock
    const result = renderFullSvg(fakeTk as any);
    expect(mockSelect).toHaveBeenCalledWith({});
    expect(mockRenderToSVG).toHaveBeenCalledWith(1);
    expect(result).toBe("<svg>full-svg</svg>");
  });
});
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd apps/web && bun vitest run src/lib/score-worker.test.ts
```

Expected: FAIL — `Cannot find module './score-worker'`

- [ ] **Step 3: Implement the minimum to make the test pass**

```typescript
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
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd apps/web && bun vitest run src/lib/score-worker.test.ts
```

Expected: PASS

- [ ] **Step 5: Commit**

```bash
cd apps/web && git add src/lib/score-worker.ts src/lib/score-worker.test.ts && git commit -m "feat(web): add Verovio score-worker with renderClipSvg + renderFullSvg"
```

---

### Task 2: Sandbox resize behavior variants

**Group:** A (parallel with Task 1)

**Behavior being verified:** Visual validation only — three panel resize behaviors rendered side-by-side in the sandbox for the developer to evaluate. No automated test.

**Files:**
- Modify: `apps/web/src/routes/app.sandbox.tsx`

---

- [ ] **Step 1: Add ResizeSandbox component to `app.sandbox.tsx`**

Add the following component and section. It uses a static inline SVG as a placeholder score (no Verovio dependency). Each variant uses a 300px-wide initial panel with a draggable right edge.

Add above the `ArtifactSandbox` function:

```typescript
// --- Resize behavior sandbox ---

const PLACEHOLDER_SVG = `<svg xmlns="http://www.w3.org/2000/svg" width="600" height="120" viewBox="0 0 600 120">
  <rect width="600" height="120" fill="#fff"/>
  <line x1="0" y1="60" x2="600" y2="60" stroke="#333" stroke-width="1"/>
  <text x="10" y="20" font-size="11" fill="#666">Placeholder score — resize panel to compare variants</text>
  <rect x="20" y="40" width="40" height="40" rx="2" fill="none" stroke="#444" stroke-width="1"/>
  <rect x="80" y="40" width="40" height="40" rx="2" fill="none" stroke="#444" stroke-width="1"/>
  <rect x="140" y="40" width="40" height="40" rx="2" fill="none" stroke="#444" stroke-width="1"/>
  <rect x="200" y="40" width="40" height="40" rx="2" fill="none" stroke="#444" stroke-width="1"/>
</svg>`;

function ResizeVariantPanel({
  label,
  description,
  svgMarkup,
  onResize,
}: {
  label: string;
  description: string;
  svgMarkup: string;
  onResize: (newWidth: number) => void;
}) {
  const [width, setWidth] = useState(300);
  const panelRef = useRef<HTMLDivElement>(null);

  function handleDragStart(e: React.MouseEvent) {
    e.preventDefault();
    const startX = e.clientX;
    const startWidth = width;

    function onMove(ev: MouseEvent) {
      const newWidth = Math.max(160, Math.min(520, startWidth + (ev.clientX - startX)));
      if (panelRef.current) panelRef.current.style.width = `${newWidth}px`;
      onResize(newWidth);
    }

    function onUp(ev: MouseEvent) {
      const newWidth = Math.max(160, Math.min(520, startWidth + (ev.clientX - startX)));
      setWidth(newWidth);
      document.removeEventListener("mousemove", onMove);
      document.removeEventListener("mouseup", onUp);
      document.body.style.cursor = "";
    }

    document.addEventListener("mousemove", onMove);
    document.addEventListener("mouseup", onUp);
    document.body.style.cursor = "col-resize";
  }

  return (
    <div className="flex flex-col gap-2">
      <div className="flex items-center gap-2">
        <span className="text-label-sm text-accent font-mono">{label}</span>
        <span className="text-body-xs text-text-tertiary">{description}</span>
      </div>
      <div
        ref={panelRef}
        className="relative border border-border rounded-lg overflow-hidden bg-white"
        style={{ width }}
      >
        <div
          // biome-ignore lint/security/noDomManipulation: controlled SVG from Verovio, not user input
          // eslint-disable-next-line react/no-unknown-property
          ref={(el) => { if (el) el.insertAdjacentHTML("afterbegin", svgMarkup); }}
          className="[&>svg]:w-full [&>svg]:block"
        />
        <div
          onMouseDown={handleDragStart}
          className="absolute right-0 top-0 bottom-0 w-1.5 cursor-col-resize bg-border hover:bg-accent transition-colors"
        />
      </div>
      <span className="text-body-xs text-text-tertiary">Width: {width}px</span>
    </div>
  );
}

function ResizeSandbox() {
  // Variant A: reflow on drag-end — score re-fetched/re-rendered after drag releases
  // Simulated here: SVG re-injected at new width on drag-end
  const [variantASvg, setVariantASvg] = useState(PLACEHOLDER_SVG);

  // Variant B: debounced reflow — re-render fires 200ms after drag stops
  const [variantBSvg, setVariantBSvg] = useState(PLACEHOLDER_SVG);
  const debounceRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  // Variant C: fixed-width CSS scale — SVG rendered once, CSS width:100% scales it
  // No state needed — the SVG is always the same markup

  function handleVariantAResize(newWidth: number) {
    // On drag-end (this is called on mouseup via setWidth), re-inject SVG at new width
    const updatedSvg = PLACEHOLDER_SVG.replace('width="600"', `width="${newWidth * 2}"`);
    setVariantASvg(updatedSvg);
  }

  function handleVariantBResize(newWidth: number) {
    if (debounceRef.current) clearTimeout(debounceRef.current);
    debounceRef.current = setTimeout(() => {
      const updatedSvg = PLACEHOLDER_SVG.replace('width="600"', `width="${newWidth * 2}"`);
      setVariantBSvg(updatedSvg);
    }, 200);
  }

  return (
    <div className="border border-border rounded-xl bg-surface-card p-5 flex flex-col gap-6">
      <div>
        <h2 className="font-display text-display-xs text-cream">Panel Resize Behavior</h2>
        <p className="text-body-sm text-text-secondary mt-1">
          Drag the right edge of each panel. Pick the behavior that feels right.
        </p>
      </div>
      <div className="flex flex-col gap-6">
        <ResizeVariantPanel
          label="A: Reflow on drag-end"
          description="Score re-renders once after you release the handle"
          svgMarkup={variantASvg}
          onResize={handleVariantAResize}
        />
        <ResizeVariantPanel
          label="B: Debounced 200ms"
          description="Score re-renders 200ms after drag stops moving"
          svgMarkup={variantBSvg}
          onResize={handleVariantBResize}
        />
        <ResizeVariantPanel
          label="C: Fixed-width CSS scale"
          description="Score rendered once at fixed width; container scales via CSS"
          svgMarkup={PLACEHOLDER_SVG}
          onResize={() => {}}
        />
      </div>
    </div>
  );
}
```

Add `useRef` to the existing import if not already present (it is). Add `ResizeSandbox` as the first section in `ArtifactSandbox`'s returned JSX, before the exercise sections:

```typescript
{/* Resize behavior sandbox — pick A, B, or C before Task 8 executes */}
<ResizeSandbox />
```

- [ ] **Step 2: Start dev server, navigate to `/app/sandbox`, test all three resize variants manually**

```bash
cd apps/web && bun run dev
```

Open `http://localhost:3000/app/sandbox`. Drag each panel's right edge. Confirm: A reflows on release, B reflows after a pause, C scales continuously without re-render.

- [ ] **Step 3: Commit**

```bash
cd apps/web && git add src/routes/app.sandbox.tsx && git commit -m "feat(web): add resize behavior sandbox variants A/B/C for score panel"
```

---

### Task 3: score-renderer.ts — basic getClip

**Group:** B (after Task 1)

**Behavior being verified:** `scoreRenderer.getClip()` resolves with the SVG string returned by the Worker.

**Interface under test:** `scoreRenderer.getClip(pieceId, startBar, endBar): Promise<string>`

**Files:**
- Create: `apps/web/src/lib/score-renderer.ts`
- Create: `apps/web/src/lib/score-renderer.test.ts`

---

- [ ] **Step 1: Write the failing test**

```typescript
// apps/web/src/lib/score-renderer.test.ts
import { beforeEach, describe, expect, it, vi } from "vitest";

class MockWorker {
  onmessage: ((e: MessageEvent) => void) | null = null;
  postMessage = vi.fn((msg: { requestId: string }) => {
    const handler = this.onmessage;
    Promise.resolve().then(() => {
      handler?.({
        data: { requestId: msg.requestId, svg: "<svg>mock</svg>" },
      } as MessageEvent);
    });
  });
  terminate = vi.fn();
}

vi.stubGlobal("Worker", MockWorker);

const mockGetData = vi.fn().mockResolvedValue(new ArrayBuffer(8));
vi.mock("./api", () => ({
  api: {
    scores: {
      getData: (...args: unknown[]) => mockGetData(...args),
    },
  },
}));

beforeEach(() => {
  vi.clearAllMocks();
  vi.resetModules();
});

describe("scoreRenderer.getClip", () => {
  it("resolves with the SVG string returned by the Worker", async () => {
    const { scoreRenderer } = await import("./score-renderer");
    const svg = await scoreRenderer.getClip("chopin.ballades.1", 1, 4);
    expect(svg).toBe("<svg>mock</svg>");
    expect(mockGetData).toHaveBeenCalledWith("chopin.ballades.1");
  });
});
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd apps/web && bun vitest run src/lib/score-renderer.test.ts
```

Expected: FAIL — `Cannot find module './score-renderer'`

- [ ] **Step 3: Implement the minimum to make the test pass**

Only `getClip` is implemented here. `getFull` is added in Task 5. Deduplication is added in Task 4. Error-retry behavior is added in Task 6.

```typescript
// apps/web/src/lib/score-renderer.ts
import { api } from "./api";

class ScoreRenderer {
  private worker: Worker;
  private pendingRequests = new Map<
    string,
    { resolve: (svg: string) => void; reject: (err: Error) => void }
  >();
  private bytesCache = new Map<string, ArrayBuffer>();
  private sentPieceIds = new Set<string>();
  private requestCounter = 0;

  constructor() {
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

  private async ensureBytes(pieceId: string): Promise<void> {
    if (this.sentPieceIds.has(pieceId) || this.bytesCache.has(pieceId)) return;
    const bytes = await api.scores.getData(pieceId);
    this.bytesCache.set(pieceId, bytes);
  }

  async getClip(pieceId: string, startBar: number, endBar: number): Promise<string> {
    await this.ensureBytes(pieceId);
    return new Promise((resolve, reject) => {
      const requestId = `req-${++this.requestCounter}`;
      this.pendingRequests.set(requestId, { resolve, reject });
      const bytes = !this.sentPieceIds.has(pieceId)
        ? this.bytesCache.get(pieceId)
        : undefined;
      if (!this.sentPieceIds.has(pieceId)) this.sentPieceIds.add(pieceId);
      this.worker.postMessage({
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
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd apps/web && bun vitest run src/lib/score-renderer.test.ts
```

Expected: PASS

- [ ] **Step 5: Commit**

```bash
cd apps/web && git add src/lib/score-renderer.ts src/lib/score-renderer.test.ts && git commit -m "feat(web): add ScoreRenderer with getClip/getFull Worker interface"
```

---

### Task 4: score-renderer.ts — fetch deduplication

**Group:** C (after Task 3)

**Behavior being verified:** Concurrent `getClip` calls for the same `pieceId` before the first fetch resolves trigger exactly one `api.scores.getData` call.

**Interface under test:** `scoreRenderer.getClip(pieceId, startBar, endBar): Promise<string>` called twice concurrently.

**Files:**
- Modify: `apps/web/src/lib/score-renderer.ts`
- Modify: `apps/web/src/lib/score-renderer.test.ts`

---

- [ ] **Step 1: Write the failing test**

Add to `apps/web/src/lib/score-renderer.test.ts` (append inside the `describe("scoreRenderer.getClip")` block):

```typescript
  it("concurrent calls for the same pieceId trigger exactly one getData fetch", async () => {
    const { scoreRenderer } = await import("./score-renderer");
    const [svg1, svg2] = await Promise.all([
      scoreRenderer.getClip("chopin.ballades.1", 1, 4),
      scoreRenderer.getClip("chopin.ballades.1", 5, 8),
    ]);
    expect(mockGetData).toHaveBeenCalledTimes(1);
    expect(svg1).toBe("<svg>mock</svg>");
    expect(svg2).toBe("<svg>mock</svg>");
  });
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd apps/web && bun vitest run src/lib/score-renderer.test.ts
```

Expected: FAIL — `mockGetData` called 2 times, expected 1

- [ ] **Step 3: Implement the minimum to make the test pass**

Replace the `ensureBytes` method in `apps/web/src/lib/score-renderer.ts` and add a `pendingFetches` class field. This version deletes the pending promise only on success — Task 6 will add the `finally` block to handle the error-retry case.

Add the class field:
```typescript
  private pendingFetches = new Map<string, Promise<void>>();
```

Replace `ensureBytes`:
```typescript
  private async ensureBytes(pieceId: string): Promise<void> {
    if (this.sentPieceIds.has(pieceId) || this.bytesCache.has(pieceId)) return;

    const inflight = this.pendingFetches.get(pieceId);
    if (inflight) return inflight;

    const fetchPromise = (async () => {
      const bytes = await api.scores.getData(pieceId);
      this.bytesCache.set(pieceId, bytes);
    })();

    this.pendingFetches.set(pieceId, fetchPromise);
    await fetchPromise;
    this.pendingFetches.delete(pieceId);
  }
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd apps/web && bun vitest run src/lib/score-renderer.test.ts
```

Expected: PASS (both tests)

- [ ] **Step 5: Commit**

```bash
cd apps/web && git add src/lib/score-renderer.ts src/lib/score-renderer.test.ts && git commit -m "feat(web): deduplicate concurrent MXL fetches in ScoreRenderer"
```

---

### Task 5: score-renderer.ts — getFull

**Group:** D (after Task 4)

**Behavior being verified:** `scoreRenderer.getFull()` resolves with the SVG string and calls `api.scores.getData` with the correct pieceId.

**Interface under test:** `scoreRenderer.getFull(pieceId): Promise<string>`

**Files:**
- Modify: `apps/web/src/lib/score-renderer.test.ts`

---

- [ ] **Step 1: Write the failing test**

Add to `apps/web/src/lib/score-renderer.test.ts`:

```typescript
describe("scoreRenderer.getFull", () => {
  it("resolves with the SVG string returned by the Worker", async () => {
    const { scoreRenderer } = await import("./score-renderer");
    const svg = await scoreRenderer.getFull("chopin.ballades.1");
    expect(svg).toBe("<svg>mock</svg>");
    expect(mockGetData).toHaveBeenCalledWith("chopin.ballades.1");
  });
});
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd apps/web && bun vitest run src/lib/score-renderer.test.ts
```

Expected: FAIL — `scoreRenderer.getFull is not a function`

- [ ] **Step 3: Implement the minimum to make the test pass**

Add the `getFull` method to the `ScoreRenderer` class in `apps/web/src/lib/score-renderer.ts`:

```typescript
  async getFull(pieceId: string): Promise<string> {
    await this.ensureBytes(pieceId);
    return new Promise((resolve, reject) => {
      const requestId = `req-${++this.requestCounter}`;
      this.pendingRequests.set(requestId, { resolve, reject });
      const bytes = !this.sentPieceIds.has(pieceId)
        ? this.bytesCache.get(pieceId)
        : undefined;
      if (!this.sentPieceIds.has(pieceId)) this.sentPieceIds.add(pieceId);
      this.worker.postMessage({ type: "render_full", requestId, pieceId, bytes });
    });
  }
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd apps/web && bun vitest run src/lib/score-renderer.test.ts
```

Expected: PASS (all tests)

- [ ] **Step 5: Commit**

```bash
cd apps/web && git add src/lib/score-renderer.ts src/lib/score-renderer.test.ts && git commit -m "feat(web): add ScoreRenderer.getFull for full-score panel rendering"
```

---

### Task 6: score-renderer.ts — error retry behavior

**Group:** E (after Task 5)

**Behavior being verified:** After a fetch failure, a subsequent `getClip` call for the same pieceId starts a fresh fetch (does not await the previously-rejected promise). This drives the `finally` block in `ensureBytes`.

**Interface under test:** `scoreRenderer.getClip(pieceId, startBar, endBar): Promise<string>` called twice — once failing, once succeeding.

**Files:**
- Modify: `apps/web/src/lib/score-renderer.ts`
- Modify: `apps/web/src/lib/score-renderer.test.ts`

---

- [ ] **Step 1: Write the failing test**

Add to `apps/web/src/lib/score-renderer.test.ts`:

```typescript
describe("scoreRenderer error retry", () => {
  it("allows a fresh fetch after the previous fetch for the same pieceId failed", async () => {
    mockGetData.mockRejectedValueOnce(new Error("network error"));
    const { scoreRenderer } = await import("./score-renderer");

    // First call fails
    await expect(
      scoreRenderer.getClip("piece-retry", 1, 4),
    ).rejects.toThrow("network error");

    // Second call should trigger a NEW fetch (pendingFetches cleared on error)
    mockGetData.mockResolvedValueOnce(new ArrayBuffer(16));
    const svg = await scoreRenderer.getClip("piece-retry", 5, 8);
    expect(svg).toBe("<svg>mock</svg>");
    expect(mockGetData).toHaveBeenCalledTimes(2);
  });
});
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd apps/web && bun vitest run src/lib/score-renderer.test.ts
```

Expected: FAIL — the second call awaits the cached (rejected) promise from `pendingFetches` and rejects with the same "network error", so `mockGetData` is called only once and the retry never happens.

- [ ] **Step 3: Implement the minimum to make the test pass**

Wrap the `await fetchPromise` in `ensureBytes` with `try/finally` so the pending fetch is cleared on both success and failure:

```typescript
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
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd apps/web && bun vitest run src/lib/score-renderer.test.ts
```

Expected: PASS (all tests in file)

- [ ] **Step 5: Commit**

```bash
cd apps/web && git add src/lib/score-renderer.ts src/lib/score-renderer.test.ts && git commit -m "feat(web): clear pendingFetches on error so retries trigger fresh fetches"
```

---

### Task 7: ScoreHighlightCard — swap osmdManager for scoreRenderer

**Group:** F (parallel with Task 8)

**Behavior being verified:** When `scoreRenderer.getClip` rejects, `ScoreHighlightCard` renders text-only fallback showing dimension, bar range, and annotation — no score visual, no crash.

**Interface under test:** `ScoreHighlightCard` React component rendered with a `ScoreHighlightConfig`.

**Files:**
- Modify: `apps/web/src/components/cards/ScoreHighlightCard.tsx`
- Modify: `apps/web/src/components/cards/ScoreHighlightCard.test.ts` → rename to `ScoreHighlightCard.test.tsx`

---

- [ ] **Step 1: Write the failing test**

Delete `apps/web/src/components/cards/ScoreHighlightCard.test.ts` and create `apps/web/src/components/cards/ScoreHighlightCard.test.tsx`:

```typescript
// apps/web/src/components/cards/ScoreHighlightCard.test.tsx
import { render, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import * as React from "react";
import type { ScoreHighlightConfig } from "../../lib/types";

const mockGetClip = vi.fn();
vi.mock("../../lib/score-renderer", () => ({
  scoreRenderer: {
    getClip: (...args: unknown[]) => mockGetClip(...args),
  },
}));

beforeEach(() => {
  vi.clearAllMocks();
  vi.resetModules();
});

describe("ScoreHighlightCard", () => {
  const config: ScoreHighlightConfig = {
    pieceId: "chopin.ballades.1",
    highlights: [
      {
        bars: [1, 4] as [number, number],
        dimension: "dynamics",
        annotation: "hushed opening",
      },
    ],
  };

  it("renders dimension label, bar range, and annotation when getClip rejects", async () => {
    mockGetClip.mockRejectedValue(new Error("Worker unavailable"));
    const { ScoreHighlightCard } = await import("./ScoreHighlightCard");
    render(React.createElement(ScoreHighlightCard, { config }));
    await waitFor(() => {
      expect(screen.getByText("dynamics")).toBeInTheDocument();
      expect(screen.getByText(/1/)).toBeInTheDocument();
      expect(screen.getByText("hushed opening")).toBeInTheDocument();
    });
  });
});
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd apps/web && bun vitest run src/components/cards/ScoreHighlightCard.test.tsx
```

Expected: FAIL — module `../../lib/score-renderer` not found in the component (component still imports `osmd-manager`).

- [ ] **Step 3: Implement the minimum to make the test pass**

Replace `apps/web/src/components/cards/ScoreHighlightCard.tsx` with the updated implementation. Key changes:

1. Remove: `import { osmdManager } from "../../lib/osmd-manager";`
2. Add: `import { scoreRenderer } from "../../lib/score-renderer";`
3. Replace the `loadScore` async function in the `useEffect`:

```typescript
// Replace the existing loadScore useEffect body with:
async function loadScore() {
  try {
    const svgResults: string[] = [];
    for (const highlight of config.highlights) {
      const svg = await scoreRenderer.getClip(
        config.pieceId,
        highlight.bars[0],
        highlight.bars[1],
      );
      svgResults.push(svg);
    }
    if (cancelled) return;

    if (svgContainerRef.current) {
      svgContainerRef.current.textContent = "";
      for (let i = 0; i < svgResults.length; i++) {
        const highlight = config.highlights[i];
        const color =
          DIMENSION_COLORS[highlight.dimension as keyof typeof DIMENSION_COLORS] ??
          "#7a9a82";
        const wrapper = document.createElement("div");
        wrapper.style.borderRadius = "6px";
        wrapper.style.border = `1.5px solid ${color}40`;
        wrapper.style.overflow = "hidden";
        wrapper.insertAdjacentHTML("beforeend", svgResults[i]);
        const innerSvg = wrapper.querySelector("svg");
        if (innerSvg) {
          innerSvg.setAttribute("width", "100%");
          innerSvg.style.display = "block";
        }
        svgContainerRef.current.appendChild(wrapper);
      }
    }

    if (!cancelled) {
      setHasClips(svgResults.length > 0);
      setRenderState("rendered");
    }
  } catch (err) {
    console.error("ScoreHighlightCard: failed to load score", err);
    if (!cancelled) setRenderState("error");
  }
}
```

4. Remove the `clearChildren` helper function (replaced by `textContent = ""`).

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd apps/web && bun vitest run src/components/cards/ScoreHighlightCard.test.tsx
```

Expected: PASS

- [ ] **Step 5: Commit**

```bash
cd apps/web && git add src/components/cards/ScoreHighlightCard.tsx src/components/cards/ScoreHighlightCard.test.tsx && git rm src/components/cards/ScoreHighlightCard.test.ts 2>/dev/null; git commit -m "feat(web): ScoreHighlightCard uses scoreRenderer instead of osmd-manager"
```

---

### Task 8: ScorePanel — swap osmdManager for scoreRenderer.getFull()

**Group:** F (parallel with Task 7)

**Prerequisite:** Before executing this task, confirm which resize variant was chosen from the Task 2 sandbox. Default: **Variant A (reflow on drag-end)**.

**Behavior being verified:** `ScorePanel` renders without error when opened via `useScorePanelStore.openHighlight()`, using `scoreRenderer` mocked at the module boundary. The existing store integration test continues to pass.

**Interface under test:** `ScorePanel` React component driven by `useScorePanelStore`.

**Files:**
- Modify: `apps/web/src/components/ScorePanel.tsx`
- Modify: `apps/web/src/components/ScorePanel.test.ts`

---

- [ ] **Step 1: Write the failing test**

Replace `apps/web/src/components/ScorePanel.test.ts` with:

```typescript
// apps/web/src/components/ScorePanel.test.ts
import { describe, expect, it, vi } from "vitest";
import { useScorePanelStore } from "../stores/score-panel";

vi.mock("../lib/score-renderer", () => ({
  scoreRenderer: {
    getFull: vi.fn().mockResolvedValue("<svg><g class='measure'/></svg>"),
    getClip: vi.fn().mockResolvedValue("<svg/>"),
  },
}));

describe("ScorePanel", () => {
  it("reads highlightData from store when opened via openHighlight", () => {
    const store = useScorePanelStore.getState();
    store.openHighlight({
      pieceId: "piece-abc",
      highlights: [
        { bars: [4, 8] as [number, number], dimension: "dynamics", annotation: "forte" },
      ],
    });

    const state = useScorePanelStore.getState();
    expect(state.isOpen).toBe(true);
    expect(state.highlightData).not.toBeNull();
    expect(state.highlightData!.pieceId).toBe("piece-abc");
    expect(state.sessionData).toBeNull();

    store.clear();
  });

  it("ScorePanel module exports ScorePanel component", async () => {
    const mod = await import("./ScorePanel");
    expect(typeof mod.ScorePanel).toBe("function");
  });
});
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd apps/web && bun vitest run src/components/ScorePanel.test.ts
```

Expected: FAIL — `Cannot find module '../lib/score-renderer'` or mock does not match the import in `ScorePanel.tsx` (which still imports `osmd-manager`).

- [ ] **Step 3: Implement the minimum to make the test pass**

Rewrite `ScorePanelScore` in `apps/web/src/components/ScorePanel.tsx`. Key changes:

1. Remove: `import { osmdManager } from "../lib/osmd-manager";`
2. Add: `import { scoreRenderer } from "../lib/score-renderer";`
3. Remove: `osmdRef` prop from `ScorePanelScoreProps` and all OSMD-related code in `ScorePanelScore`.
4. Replace the `initOSMD` body inside `ScorePanelScore`'s `useMountEffect`:

```typescript
// Replace initOSMD with:
async function loadScore() {
  const container = containerRef.current;
  if (!container || cancelled) return;

  if (!pieceId) {
    setIsRendered(true);
    return;
  }

  try {
    const svg = await scoreRenderer.getFull(pieceId);
    if (cancelled) return;
    container.textContent = "";
    container.insertAdjacentHTML("beforeend", svg);
    setIsRendered(true);
  } catch (err) {
    console.error("ScorePanel: score render failed", err);
    if (!cancelled) setIsError(true);
  }
}
```

5. Replace annotation positioning in the `useEffect` that depends on `[isRendered, observations]`:

```typescript
useEffect(() => {
  if (!isRendered || !containerRef.current) return;

  const containerRect = containerRef.current.getBoundingClientRect();
  const measureEls = Array.from(
    containerRef.current.querySelectorAll<Element>(".measure"),
  );
  const positions: AnnotationPosition[] = [];

  for (const obs of observations) {
    if (!obs.barRange) {
      positions.push({ top: 0, left: 0 });
      continue;
    }
    const measureIdx = obs.barRange[0] - 1;
    const el = measureEls[measureIdx];
    if (el) {
      const rect = el.getBoundingClientRect();
      positions.push({
        top: rect.top - containerRect.top - 28,
        left: rect.left - containerRect.left,
      });
    } else {
      positions.push({ top: 60 + positions.length * 80, left: 20 });
    }
  }

  setAnnotationPositions(positions);
}, [isRendered, observations]);
```

6. Remove `osmdRef` from the `ScorePanelScore` component props and the ref passed from `ScorePanel`.

7. In `ScorePanel`'s drag-end handler (`onMouseUp`), replace the `osmdRef.current.render()` call with a re-render trigger. For Variant A (reflow on drag-end), dispatch a custom event or use a state increment to remount `ScorePanelScore`:

In `ScorePanel`, replace the `osmdRef` ref and add a render key:

```typescript
// Replace: const osmdRef = useRef<any>(null);
// Add:
const [scoreRenderKey, setScoreRenderKey] = useState(0);
```

In `onMouseUp`, replace `osmdRef.current.render()` with:

```typescript
setPanelWidth(dragWidthRef.current);
setScoreRenderKey((k) => k + 1); // triggers ScorePanelScore remount → re-fetch getFull
```

Update the `ScorePanelScore` key prop:

```typescript
<ScorePanelScore
  key={`${pieceId}-${title}-${observations.length}-${scoreRenderKey}`}
  ...
/>
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd apps/web && bun vitest run src/components/ScorePanel.test.ts
```

Expected: PASS

- [ ] **Step 5: Commit**

```bash
cd apps/web && git add src/components/ScorePanel.tsx src/components/ScorePanel.test.ts && git commit -m "feat(web): ScorePanel uses scoreRenderer.getFull with Verovio SVG annotation positioning"
```

---

### Task 9: Cleanup — delete osmd-manager, remove opensheetmusicdisplay

**Group:** G (after Tasks 7 and 8)

**Behavior being verified:** Full test suite passes with OSMD dependency removed. No file imports `osmd-manager`.

**Files:**
- Delete: `apps/web/src/lib/osmd-manager.ts`
- Delete: `apps/web/src/lib/osmd-manager.test.ts`
- Modify: `apps/web/package.json`

---

- [ ] **Step 1: Verify no remaining imports of osmd-manager**

```bash
grep -r "osmd-manager\|osmdManager" apps/web/src --include="*.ts" --include="*.tsx"
```

Expected: no output. If any files still import `osmd-manager`, update them to use `scoreRenderer` before proceeding.

- [ ] **Step 2: Delete osmd-manager files**

```bash
cd apps/web && rm src/lib/osmd-manager.ts src/lib/osmd-manager.test.ts
```

- [ ] **Step 3: Remove opensheetmusicdisplay from package.json**

```bash
cd apps/web && bun remove opensheetmusicdisplay
```

- [ ] **Step 4: Run full test suite — verify it PASSES**

```bash
cd apps/web && bun run test
```

Expected: PASS — all tests pass, no references to opensheetmusicdisplay or osmd-manager.

- [ ] **Step 5: Commit**

```bash
cd apps/web && git add -A && git commit -m "chore(web): remove osmd-manager and opensheetmusicdisplay dependency"
```
