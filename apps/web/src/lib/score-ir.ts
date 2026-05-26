// apps/web/src/lib/score-ir.ts

export type Bbox = { x: number; y: number; w: number; h: number };
// NOTE: w and h are always 0. SVG-text parsing has no layout engine.
// Cursor reads only x and y. Click-to-select would need main-thread getBBox.

export type NoteIR = {
  id: string;
  bbox: Bbox;
  qstamp: number;
  staff: 1 | 2;
};

export type BarIR = {
  barNumber: number;
  measureOn: string;
  pageN: number;
  bbox: Bbox;
  noteIds: string[];
  qstampStart: number;
  qstampEnd: number;
};

export type PageIR = {
  pageN: number;
  viewBox: string;
  width: number;
  height: number;
  systemBboxes: Bbox[];
};

export type ScoreIR = {
  pieceId: string;
  verovioVersion: string;
  pageWidth: number;
  pages: PageIR[];
  bars: BarIR[];
  notes: Record<string, NoteIR>;
};

interface MeasureEntry {
  qstamp: number;
  measureOn: string;
}

// Extract x or y from a <use> element's attribute string.
// These are the only two attributes we need, so we use hardcoded patterns.
function extractX(attrs: string): number {
  const m = attrs.match(/\bx="([^"]+)"/);
  return m ? parseFloat(m[1]) : Number.NaN;
}

function extractY(attrs: string): number {
  const m = attrs.match(/\by="([^"]+)"/);
  return m ? parseFloat(m[1]) : Number.NaN;
}

// Parse the viewBox attribute "minX minY width height" into width and height.
function parseViewBox(attrs: string): { viewBox: string; width: number; height: number } {
  const m = attrs.match(/viewBox="([^"]+)"/);
  const viewBox = m ? m[1] : "0 0 0 0";
  const parts = viewBox.split(/\s+/).map(Number);
  return { viewBox, width: parts[2] ?? 0, height: parts[3] ?? 0 };
}

// Walk SVG text to extract note elements: <g class="note" id="..."><use x="..." y="..."/>
// Returns a map of note id -> {x, y}.
function extractNotePositions(svgText: string): Map<string, { x: number; y: number }> {
  const result = new Map<string, { x: number; y: number }>();
  // Match <g ...class="...note..."... id="..."> followed (soon after) by <use x="..." y="..."/>
  const noteBlockRe = /<g\s+([^>]*class="[^"]*\bnote\b[^"]*"[^>]*)>([\s\S]*?)<\/g>/g;
  for (const blockMatch of svgText.matchAll(noteBlockRe)) {
    const gAttrs = blockMatch[1];
    const inner = blockMatch[2];
    const idMatch = gAttrs.match(/id="([^"]+)"/);
    if (!idMatch) continue;
    const id = idMatch[1];
    const useMatch = inner.match(/<use\s+([^>]*\/?>)/);
    if (!useMatch) continue;
    const useAttrs = useMatch[1];
    const x = extractX(useAttrs);
    const y = extractY(useAttrs);
    if (Number.isFinite(x) && Number.isFinite(y)) {
      result.set(id, { x, y });
    }
  }
  return result;
}

// Walk SVG text to extract measure element ids in document order.
// Returns an array of { measureId, noteIds[] } where noteIds are the note element
// ids contained within each measure block.
function extractMeasureNoteMap(svgText: string): Array<{ measureId: string; noteIds: string[] }> {
  const result: Array<{ measureId: string; noteIds: string[] }> = [];
  // Match outer measure blocks (non-greedy; may miss deeply nested — acceptable for v1).
  const measureRe = /<g\s+([^>]*class="[^"]*\bmeasure\b[^"]*"[^>]*)>([\s\S]*?)<\/g>/g;
  for (const measureMatch of svgText.matchAll(measureRe)) {
    const gAttrs = measureMatch[1];
    const inner = measureMatch[2];
    const idMatch = gAttrs.match(/id="([^"]+)"/);
    if (!idMatch) continue;
    const measureId = idMatch[1];
    const noteIds: string[] = [];
    const noteIdRe = /<g\s+[^>]*class="[^"]*\bnote\b[^"]*"[^>]*\sid="([^"]+)"/g;
    for (const noteMatch of inner.matchAll(noteIdRe)) {
      noteIds.push(noteMatch[1]);
    }
    result.push({ measureId, noteIds });
  }
  return result;
}

export function parseScoreIR(
  pieceId: string,
  pageSvgs: string[],
  measures: MeasureEntry[],
  noteQstampMap: Map<string, number>,
  verovioVersion: string,
  pageWidth: number,
): ScoreIR {
  const notes: Record<string, NoteIR> = {};
  const pages: PageIR[] = [];
  const bars: BarIR[] = [];

  // Build a lookup from measureOn id -> MeasureEntry index for qstamp resolution.
  const measureByMeasureOn = new Map<string, { qstamp: number; idx: number }>();
  for (let i = 0; i < measures.length; i++) {
    measureByMeasureOn.set(measures[i].measureOn, { qstamp: measures[i].qstamp, idx: i });
  }

  for (let pageIdx = 0; pageIdx < pageSvgs.length; pageIdx++) {
    const pageN = pageIdx + 1;
    const svgText = pageSvgs[pageIdx];

    // Extract page dimensions from the root <svg> tag.
    const svgTagMatch = svgText.match(/<svg\s+([^>]*)>/);
    const svgAttrs = svgTagMatch ? svgTagMatch[1] : "";
    const { viewBox, width, height } = parseViewBox(svgAttrs);

    pages.push({ pageN, viewBox, width, height, systemBboxes: [] });

    // Extract all note positions on this page.
    const notePositions = extractNotePositions(svgText);
    for (const [id, { x, y }] of notePositions) {
      // Staff inference: notes with y > page_midpoint are staff 2, else staff 1.
      const midY = height / 2;
      // Per-note qstamp from the timemap. Fall back to 0 if the note id is absent
      // (this should not occur for well-formed Verovio output but is explicit, not silent).
      const qstamp = noteQstampMap.get(id);
      if (qstamp === undefined) {
        console.error(`[score-ir] note id "${id}" absent from noteQstampMap — falling back to 0`);
      }
      notes[id] = {
        id,
        bbox: { x, y, w: 0, h: 0 },
        qstamp: qstamp ?? 0,
        staff: y > midY ? 2 : 1,
      };
    }

    // Extract measure->note mapping for this page.
    const measureNoteMap = extractMeasureNoteMap(svgText);

    for (const { measureId, noteIds } of measureNoteMap) {
      const entry = measureByMeasureOn.get(measureId);
      if (!entry) continue;

      const { qstamp: qstampStart, idx } = entry;
      // qstampEnd: next measure's qstamp, or qstampStart + 4 for the last measure.
      const nextMeasure = measures[idx + 1];
      const qstampEnd = nextMeasure ? nextMeasure.qstamp : qstampStart + 4;

      // Compute bar bbox from the x-positions of its notes.
      const xs = noteIds.map((id) => notes[id]?.bbox.x ?? 0).filter((x) => x > 0);
      const barX = xs.length > 0 ? Math.min(...xs) : 0;

      bars.push({
        barNumber: idx + 1,
        measureOn: measureId,
        pageN,
        bbox: { x: barX, y: 0, w: 0, h: 0 },
        noteIds,
        qstampStart,
        qstampEnd,
      });
    }
  }

  // Ensure bars are sorted by barNumber ascending.
  bars.sort((a, b) => a.barNumber - b.barNumber);

  return { pieceId, verovioVersion, pageWidth, pages, bars, notes };
}
