// apps/web/src/lib/score-cursor.ts
import type { BarIR, NoteIR, ScoreIR } from "./score-ir";
import { Sentry } from "./sentry";

export interface ScoreCursorOptions {
  pieceId: string;
  container: HTMLElement;
  ir: ScoreIR;
  qstampSource: () => number | null;
}

export class ScoreCursor {
  private readonly container: HTMLElement;
  private readonly ir: ScoreIR;
  private readonly qstampSource: () => number | null;
  // One overlay <svg> per page in the IR.
  private overlays: SVGSVGElement[] = [];
  private rafId: number | null = null;
  private lastPageN = -1;

  constructor(opts: ScoreCursorOptions) {
    this.container = opts.container;
    this.ir = opts.ir;
    this.qstampSource = opts.qstampSource;
  }

  start(): void {
    if (this.rafId !== null) return;
    this.mountOverlays();
    this.rafId = requestAnimationFrame(this.tick);
  }

  stop(): void {
    if (this.rafId !== null) {
      cancelAnimationFrame(this.rafId);
      this.rafId = null;
    }
    this.unmountOverlays();
  }

  private mountOverlays(): void {
    for (let i = 0; i < this.ir.pages.length; i++) {
      const page = this.ir.pages[i];
      const overlay = document.createElementNS("http://www.w3.org/2000/svg", "svg");
      overlay.setAttribute("class", "score-cursor-overlay");
      overlay.setAttribute("viewBox", page.viewBox);
      overlay.setAttribute("style", [
        "position:absolute",
        "top:0",
        "left:0",
        "width:100%",
        "height:100%",
        "pointer-events:none",
      ].join(";"));

      const line = document.createElementNS("http://www.w3.org/2000/svg", "line");
      line.setAttribute("x1", "0");
      line.setAttribute("x2", "0");
      line.setAttribute("y1", "0");
      line.setAttribute("y2", String(page.height));
      line.setAttribute("stroke", "#2563eb");
      line.setAttribute("stroke-width", "2");
      // Keep stroke width in screen pixels regardless of viewBox scaling.
      // Without this, stroke-width="2" in a 24000-unit-wide viewBox shown in
      // ~800px renders as ~0.07px — invisible.
      line.setAttribute("vector-effect", "non-scaling-stroke");
      line.setAttribute("visibility", "hidden");
      overlay.appendChild(line);

      this.container.appendChild(overlay);
      this.overlays.push(overlay);
    }
  }

  private unmountOverlays(): void {
    for (const overlay of this.overlays) {
      if (overlay.parentNode) {
        overlay.parentNode.removeChild(overlay);
      }
    }
    this.overlays = [];
  }

  private tick = (_ts: number): void => {
    // Guard: stop() may have been called before this tick ran.
    if (this.rafId === null) return;
    try {
      const q = this.qstampSource();
      if (q === null) {
        this.hideAll();
        this.rafId = requestAnimationFrame(this.tick);
        return;
      }

      const bar = this.findBar(q);
      if (!bar) {
        this.hideAll();
        this.rafId = requestAnimationFrame(this.tick);
        return;
      }

      const x = this.interpolateX(bar, q);
      const overlayIdx = bar.pageN - 1;
      const overlay = this.overlays[overlayIdx];
      if (!overlay) {
        this.rafId = requestAnimationFrame(this.tick);
        return;
      }

      // Compute the vertical range for the current bar's system from the y
      // values of notes in this bar. bar.bbox.y/h are zero in the IR (the
      // parser only fills x). Notes' bbox.y is the glyph baseline; pad above
      // and below to cover stems, beams, and ledger lines.
      const ys: number[] = [];
      for (const id of bar.noteIds) {
        const note = this.ir.notes[id];
        if (note && Number.isFinite(note.bbox.y) && note.bbox.y > 0) {
          ys.push(note.bbox.y);
        }
      }
      const SYSTEM_PAD = 400; // viewBox units; ~one staff-line spacing
      const pageHeight = this.ir.pages[overlayIdx]?.height ?? 0;
      const y1 = ys.length > 0 ? Math.max(0, Math.min(...ys) - SYSTEM_PAD) : 0;
      const y2 =
        ys.length > 0
          ? Math.min(pageHeight || Number.POSITIVE_INFINITY, Math.max(...ys) + SYSTEM_PAD)
          : pageHeight;

      // Hide all overlays except the current page.
      for (let i = 0; i < this.overlays.length; i++) {
        const line = this.overlays[i].querySelector("line");
        if (!line) continue;
        if (i === overlayIdx) {
          line.setAttribute("x1", String(x));
          line.setAttribute("x2", String(x));
          line.setAttribute("y1", String(y1));
          line.setAttribute("y2", String(y2));
          line.setAttribute("visibility", "visible");
        } else {
          line.setAttribute("visibility", "hidden");
        }
      }

      // Scroll into view when the cursor crosses a page boundary.
      if (bar.pageN !== this.lastPageN) {
        (overlay as unknown as { scrollIntoView?: (opts: ScrollIntoViewOptions) => void }).scrollIntoView?.({ block: "nearest" });
        this.lastPageN = bar.pageN;
      }
    } catch (err) {
      this.hideAll();
      Sentry.captureException(err);
    }
    this.rafId = requestAnimationFrame(this.tick);
  };

  private hideAll(): void {
    for (const overlay of this.overlays) {
      const line = overlay.querySelector("line");
      if (line) line.setAttribute("visibility", "hidden");
    }
  }

  // Binary search: find the bar where qstampStart <= q < qstampEnd.
  private findBar(q: number): BarIR | null {
    const bars = this.ir.bars;
    let lo = 0;
    let hi = bars.length - 1;
    while (lo <= hi) {
      const mid = (lo + hi) >>> 1;
      const bar = bars[mid];
      if (q < bar.qstampStart) {
        hi = mid - 1;
      } else if (q >= bar.qstampEnd) {
        lo = mid + 1;
      } else {
        return bar;
      }
    }
    // q is at or past the last bar's qstampEnd — return the last bar.
    return bars[bars.length - 1] ?? null;
  }

  // Linear interpolation of x within a bar between the two bracketing notes.
  private interpolateX(bar: BarIR, q: number): number {
    const notes = bar.noteIds
      .map((id) => this.ir.notes[id])
      .filter((n): n is NoteIR => n !== undefined && !Number.isNaN(n.qstamp))
      .sort((a, b) => a.qstamp - b.qstamp);

    if (notes.length === 0) return bar.bbox.x;
    if (notes.length === 1) return notes[0].bbox.x;

    // Find the two bracketing notes.
    let prev = notes[0];
    let next = notes[notes.length - 1];
    for (let i = 0; i < notes.length - 1; i++) {
      if (notes[i].qstamp <= q && notes[i + 1].qstamp > q) {
        prev = notes[i];
        next = notes[i + 1];
        break;
      }
    }

    if (prev === next) return prev.bbox.x;
    const t = (q - prev.qstamp) / (next.qstamp - prev.qstamp);
    return prev.bbox.x + t * (next.bbox.x - prev.bbox.x);
  }
}
