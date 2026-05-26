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

// Extract x and y from a <use> element's attribute string.
// Verovio uses transform="translate(x, y) scale(...)" — we parse the translate values.
// Falls back to x="..." y="..." attributes (used in synthetic test fixtures).
function extractTranslateXY(attrs: string): { x: number; y: number } {
  const tm = attrs.match(/transform="translate\(\s*([-\d.]+)\s*,\s*([-\d.]+)\s*\)/);
  if (tm) return { x: parseFloat(tm[1]), y: parseFloat(tm[2]) };
  // Fallback for plain x/y attributes (synthetic SVGs in tests).
  const xm = attrs.match(/\bx="([-\d.]+)"/);
  const ym = attrs.match(/\by="([-\d.]+)"/);
  return {
    x: xm ? parseFloat(xm[1]) : Number.NaN,
    y: ym ? parseFloat(ym[1]) : Number.NaN,
  };
}

// Parse the viewBox attribute "minX minY width height" into width and height.
function parseViewBox(attrs: string): { viewBox: string; width: number; height: number } {
  const m = attrs.match(/viewBox="([^"]+)"/);
  const viewBox = m ? m[1] : "0 0 0 0";
  const parts = viewBox.split(/\s+/).map(Number);
  return { viewBox, width: parts[2] ?? 0, height: parts[3] ?? 0 };
}

// Walk SVG text to extract note positions.
// For each note <g> tag, find the next <use> tag in the SVG text and parse its position.
// Uses a linear scan to avoid nested-</g> ambiguity.
function extractNotePositions(svgText: string): Map<string, { x: number; y: number }> {
  const result = new Map<string, { x: number; y: number }>();

  const gTagRe = /<g\s+([^>]*)>/g;
  const useTagRe = /<use\s+([^>]*\/?>)/g;

  // Collect note tag positions.
  type NoteTag = { pos: number; endPos: number; id: string };
  const noteTags: NoteTag[] = [];
  for (const m of svgText.matchAll(gTagRe)) {
    const attrs = m[1];
    if (!/class="[^"]*\bnote\b/.test(attrs)) continue;
    const idMatch = attrs.match(/\bid="([^"]+)"/);
    if (!idMatch) continue;
    noteTags.push({ pos: m.index ?? 0, endPos: (m.index ?? 0) + m[0].length, id: idMatch[1] });
  }

  // Collect use tag positions.
  type UseTag = { pos: number; attrs: string };
  const useTags: UseTag[] = [];
  for (const m of svgText.matchAll(useTagRe)) {
    useTags.push({ pos: m.index ?? 0, attrs: m[1] });
  }

  // For each note tag, find the first use tag that appears after it.
  let useIdx = 0;
  for (const note of noteTags) {
    // Advance useIdx to the first <use> that starts after this note's opening tag.
    while (useIdx < useTags.length && useTags[useIdx].pos < note.endPos) {
      useIdx++;
    }
    if (useIdx >= useTags.length) break;
    const { x, y } = extractTranslateXY(useTags[useIdx].attrs);
    if (Number.isFinite(x) && Number.isFinite(y)) {
      result.set(note.id, { x, y });
    }
  }

  return result;
}

// Walk SVG text to build a measure -> noteIds mapping.
// Strategy: scan linearly for measure and note opening tags in document order.
// When we see a measure tag, we start collecting note ids.
// When we see the next measure tag, we close the previous one.
// This avoids nested-</g> regex ambiguity entirely.
function extractMeasureNoteMap(svgText: string): Array<{ measureId: string; noteIds: string[] }> {
  const result: Array<{ measureId: string; noteIds: string[] }> = [];

  // Collect positions of all measure and note opening tags with their ids.
  type TagEvent = { pos: number; type: "measure" | "note"; id: string };
  const events: TagEvent[] = [];

  const gTagRe = /<g\s+([^>]*)>/g;
  for (const m of svgText.matchAll(gTagRe)) {
    const attrs = m[1];
    const idMatch = attrs.match(/\bid="([^"]+)"/);
    if (!idMatch) continue;
    const id = idMatch[1];
    if (/class="[^"]*\bmeasure\b/.test(attrs)) {
      events.push({ pos: m.index ?? 0, type: "measure", id });
    } else if (/class="[^"]*\bnote\b/.test(attrs)) {
      events.push({ pos: m.index ?? 0, type: "note", id });
    }
  }

  // Group notes to their containing measure by document position.
  let currentMeasure: { measureId: string; noteIds: string[] } | null = null;
  for (const ev of events) {
    if (ev.type === "measure") {
      if (currentMeasure) result.push(currentMeasure);
      currentMeasure = { measureId: ev.id, noteIds: [] };
    } else if (ev.type === "note" && currentMeasure) {
      currentMeasure.noteIds.push(ev.id);
    }
  }
  if (currentMeasure) result.push(currentMeasure);

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
      // Per-note qstamp from the timemap. Use NaN when the note id is absent so
      // the miss is structurally distinguishable from a real qstamp=0 downbeat note.
      // Consumers (ScoreCursor.interpolateX) must filter NaN notes out of interpolation.
      const qstamp = noteQstampMap.get(id);
      if (qstamp === undefined) {
        console.error(`[score-ir] note id "${id}" absent from noteQstampMap — recording NaN qstamp`);
      }
      notes[id] = {
        id,
        bbox: { x, y, w: 0, h: 0 },
        qstamp: qstamp ?? NaN,
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
