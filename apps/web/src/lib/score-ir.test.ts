// apps/web/src/lib/score-ir.test.ts
import { describe, expect, it } from "vitest";

// Minimal synthetic SVG that exercises the regex paths.
// Two measures, two notes each — one per staff.
const SYNTHETIC_PAGE_SVG = `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 2400 800" width="2400" height="800">
  <g class="measure" id="m1">
    <g class="note" id="note-1-1"><use x="100" y="200"/></g>
    <g class="note" id="note-1-2"><use x="200" y="200"/></g>
  </g>
  <g class="measure" id="m2">
    <g class="note" id="note-2-1"><use x="400" y="200"/></g>
    <g class="note" id="note-2-2"><use x="500" y="200"/></g>
  </g>
</svg>`;

const SYNTHETIC_MEASURES = [
  { qstamp: 0, measureOn: "m1" },
  { qstamp: 4, measureOn: "m2" },
];

// noteQstampMap: per-note onset qstamps from the timemap.
// m1 has two notes at different onsets (0 and 2); m2 has two notes at different onsets (4 and 6).
const SYNTHETIC_NOTE_QSTAMP_MAP = new Map<string, number>([
  ["note-1-1", 0],
  ["note-1-2", 2],
  ["note-2-1", 4],
  ["note-2-2", 6],
]);

describe("parseScoreIR — structural invariants", () => {
  it("returns a ScoreIR whose bars.length equals the supplied measures count", async () => {
    const { parseScoreIR } = await import("./score-ir");
    const ir = parseScoreIR(
      "test-piece",
      [SYNTHETIC_PAGE_SVG],
      SYNTHETIC_MEASURES,
      SYNTHETIC_NOTE_QSTAMP_MAP,
      "4.0.0",
      2400,
    );
    expect(ir.bars.length).toBe(SYNTHETIC_MEASURES.length);
  });

  it("every NoteIR.bbox.x and .y is a finite number", async () => {
    const { parseScoreIR } = await import("./score-ir");
    const ir = parseScoreIR(
      "test-piece",
      [SYNTHETIC_PAGE_SVG],
      SYNTHETIC_MEASURES,
      SYNTHETIC_NOTE_QSTAMP_MAP,
      "4.0.0",
      2400,
    );
    for (const note of Object.values(ir.notes)) {
      expect(Number.isFinite(note.bbox.x)).toBe(true);
      expect(Number.isFinite(note.bbox.y)).toBe(true);
      expect(note.bbox.w).toBe(0);
      expect(note.bbox.h).toBe(0);
    }
  });

  it("every BarIR.noteIds resolves to a key in ir.notes", async () => {
    const { parseScoreIR } = await import("./score-ir");
    const ir = parseScoreIR(
      "test-piece",
      [SYNTHETIC_PAGE_SVG],
      SYNTHETIC_MEASURES,
      SYNTHETIC_NOTE_QSTAMP_MAP,
      "4.0.0",
      2400,
    );
    for (const bar of ir.bars) {
      for (const noteId of bar.noteIds) {
        expect(ir.notes[noteId]).toBeDefined();
      }
    }
  });

  it("qstampStart < qstampEnd for every bar with notes", async () => {
    const { parseScoreIR } = await import("./score-ir");
    const ir = parseScoreIR(
      "test-piece",
      [SYNTHETIC_PAGE_SVG],
      SYNTHETIC_MEASURES,
      SYNTHETIC_NOTE_QSTAMP_MAP,
      "4.0.0",
      2400,
    );
    for (const bar of ir.bars) {
      if (bar.noteIds.length > 0) {
        expect(bar.qstampStart).toBeLessThan(bar.qstampEnd);
      }
    }
  });

  it("notes with distinct onset positions carry distinct qstamp values (not all qstampStart)", async () => {
    const { parseScoreIR } = await import("./score-ir");
    const ir = parseScoreIR(
      "test-piece",
      [SYNTHETIC_PAGE_SVG],
      SYNTHETIC_MEASURES,
      SYNTHETIC_NOTE_QSTAMP_MAP,
      "4.0.0",
      2400,
    );
    // m1 has notes at qstamp 0 and 2 — they must be distinct in the IR.
    const bar1 = ir.bars.find((b) => b.measureOn === "m1");
    expect(bar1).toBeDefined();
    if (bar1 && bar1.noteIds.length >= 2) {
      const qstamps = bar1.noteIds.map((id) => ir.notes[id]?.qstamp ?? -1);
      const uniqueQstamps = new Set(qstamps);
      expect(uniqueQstamps.size).toBeGreaterThan(1);
    }
  });

  it("stores pieceId, verovioVersion, and pageWidth on the IR", async () => {
    const { parseScoreIR } = await import("./score-ir");
    const ir = parseScoreIR(
      "my-piece",
      [SYNTHETIC_PAGE_SVG],
      SYNTHETIC_MEASURES,
      SYNTHETIC_NOTE_QSTAMP_MAP,
      "4.1.0",
      2400,
    );
    expect(ir.pieceId).toBe("my-piece");
    expect(ir.verovioVersion).toBe("4.1.0");
    expect(ir.pageWidth).toBe(2400);
  });

  it("note absent from noteQstampMap gets NaN qstamp (not 0)", async () => {
    const { parseScoreIR } = await import("./score-ir");
    // Omit note-1-2 from the map so it is absent during parsing.
    const partialMap = new Map<string, number>([
      ["note-1-1", 0],
      // note-1-2 deliberately absent
      ["note-2-1", 4],
      ["note-2-2", 6],
    ]);
    const ir = parseScoreIR(
      "test-piece",
      [SYNTHETIC_PAGE_SVG],
      SYNTHETIC_MEASURES,
      partialMap,
      "4.0.0",
      2400,
    );
    const missedNote = ir.notes["note-1-2"];
    expect(missedNote).toBeDefined();
    expect(Number.isNaN(missedNote?.qstamp)).toBe(true);
    // The present note at the real downbeat (qstamp=0) must NOT be NaN.
    expect(Number.isNaN(ir.notes["note-1-1"]?.qstamp)).toBe(false);
    expect(ir.notes["note-1-1"]?.qstamp).toBe(0);
  });
});
