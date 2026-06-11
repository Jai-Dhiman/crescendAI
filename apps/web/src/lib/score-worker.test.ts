// apps/web/src/lib/score-worker.test.ts
import { beforeEach, describe, expect, it, vi } from "vitest";

const mockGetPageWithElement = vi.fn().mockReturnValue(3);
const mockRenderToSVG = vi.fn().mockReturnValue("<svg>clip-svg</svg>");
const mockRenderToTimemap = vi.fn().mockReturnValue([]);
const mockLoadZipDataBuffer = vi.fn();

const fakeTk = {
	renderToSVG: mockRenderToSVG,
	renderToTimemap: mockRenderToTimemap,
	getPageWithElement: mockGetPageWithElement,
	loadZipDataBuffer: mockLoadZipDataBuffer,
	setOptions: vi.fn(),
};

const fakeMeasures = [
	{ qstamp: 0, measureOn: "measure-id-1" },
	{ qstamp: 4, measureOn: "measure-id-2" },
	{ qstamp: 8, measureOn: "measure-id-3" },
	{ qstamp: 12, measureOn: "measure-id-4" },
	{ qstamp: 16, measureOn: "measure-id-5" },
];

beforeEach(() => {
	vi.clearAllMocks();
	mockGetPageWithElement.mockReturnValue(3);
	mockRenderToSVG.mockReturnValue("<svg>clip-svg</svg>");
});

describe("processRenderClipRequest", () => {
	// NOTE: these are dispatch-shape unit tests. Real cropping behaviour against
	// Verovio is verified in score-worker.integration.test.ts — which is what
	// actually proves the SVG output is cropped to the requested bar range.
	it("calls tk.select with measureRange and redoLayout before rendering", async () => {
		const mockSelect = vi.fn();
		const mockRedoLayout = vi.fn();
		const tk = {
			...fakeTk,
			select: mockSelect,
			redoLayout: mockRedoLayout,
		};
		// Use a recognizable marker so the .toBe assertion proves the function
		// returns the renderToSVG output unchanged, not some transformed value.
		mockRenderToSVG.mockReturnValue('<svg data-cropped="2-4">clip-svg</svg>');
		const { processRenderClipRequest } = await import("./score-worker");
		// biome-ignore lint/suspicious/noExplicitAny: test mock
		const svg = processRenderClipRequest(tk as any, fakeMeasures, 2, 4);
		expect(mockSelect).toHaveBeenCalledWith({ measureRange: "2-4" });
		expect(mockRedoLayout).toHaveBeenCalled();
		expect(mockRenderToSVG).toHaveBeenCalledWith(1);
		expect(svg).toBe('<svg data-cropped="2-4">clip-svg</svg>');
	});

	it("falls back to a full page-1 render when the start bar is out of range", async () => {
		const tk = { ...fakeTk, select: vi.fn(), redoLayout: vi.fn() };
		const { processRenderClipRequest } = await import("./score-worker");
		// biome-ignore lint/suspicious/noExplicitAny: test mock
		const svg = processRenderClipRequest(tk as any, fakeMeasures, 999, 1000);
		expect(tk.select).not.toHaveBeenCalled();
		expect(mockRenderToSVG).toHaveBeenCalledWith(1);
		expect(svg).toBe("<svg>clip-svg</svg>");
	});
});


describe("processGetPageRequest", () => {
  it("returns the pre-rendered SVG for the requested page number", async () => {
    const { processGetPageRequest } = await import("./score-worker");
    const pageSvgs = ["<svg>page1</svg>", "<svg>page2</svg>", "<svg>page3</svg>"];
    const result = processGetPageRequest(pageSvgs, 2);
    expect(result).toBe("<svg>page2</svg>");
  });

  it("returns 'failed' when the requested page does not exist", async () => {
    const { processGetPageRequest } = await import("./score-worker");
    const pageSvgs = ["<svg>page1</svg>"];
    const result = processGetPageRequest(pageSvgs, 99);
    expect(result).toBe("failed");
  });
});

describe("processGetClipPlaybackRequest", () => {
  it("returns svg, clip-scoped ir, and notes array from timemap + MIDI values", async () => {
    const mockSelect = vi.fn();
    const mockRedoLayout = vi.fn();
    const mockGetVersion = vi.fn().mockReturnValue("4.0.0");
    const mockGetPageCount = vi.fn().mockReturnValue(1);
    // Timemap: two notes at qstamps 0 and 2, plus a measureOn.
    // measureOn id "measure-id-1" must match the measure class attribute in the SVG below
    // so that parseScoreIR can build bars from the SVG.
    const mockTimemap = [
      { qstamp: 0, measureOn: "measure-id-1", on: ["note-a"] },
      { qstamp: 2, on: ["note-b"] },
      { qstamp: 4, measureOn: "measure-id-2", on: [] },
    ];
    // Synthetic SVG containing one measure element with id="measure-id-1" and one
    // note element with id="note-a". parseScoreIR scans for class="... measure ..."
    // and class="... note ..." attributes with matching ids. The <use> tag supplies
    // the x/y position for the note bbox (plain x= y= attributes, supported by the
    // extractTranslateXY fallback).
    const clipSvg = `<svg viewBox="0 0 1600 800">
      <g class="measure" id="measure-id-1">
        <g class="note" id="note-a"><use x="100" y="200"/></g>
      </g>
    </svg>`;
    // Each note has a MIDI value
    const mockGetMIDIValuesForElement = vi.fn((id: string) =>
      id === "note-a" ? [{ pitch: 60 }] : [{ pitch: 64 }]
    );
    const tk = {
      ...fakeTk,
      select: mockSelect,
      redoLayout: mockRedoLayout,
      getVersion: mockGetVersion,
      getPageCount: mockGetPageCount,
      renderToSVG: vi.fn().mockReturnValue(clipSvg),
      renderToTimemap: vi.fn().mockReturnValue(mockTimemap),
      getMIDIValuesForElement: mockGetMIDIValuesForElement,
    };

    const { processGetClipPlaybackRequest } = await import("./score-worker");
    // biome-ignore lint/suspicious/noExplicitAny: test mock
    const result = await processGetClipPlaybackRequest(tk as any, fakeMeasures, 1, 2);

    // result must be the resolved value, not a Promise
    expect(result).not.toBe("failed");
    if (result === "failed") return;

    // Concrete field assertions
    expect(typeof result.svg).toBe("string");
    expect(result.svg.length).toBeGreaterThan(0);
    expect(Array.isArray(result.ir.bars)).toBe(true);
    expect(result.ir.bars.length).toBeGreaterThan(0);
    expect(Array.isArray(result.notes)).toBe(true);
    // The timemap supplies note-a at qstamp 0 with MIDI pitch 60.
    expect(result.notes.length).toBeGreaterThan(0);
    const n = result.notes[0];
    expect(typeof n.midi).toBe("number");
    expect(typeof n.startQ).toBe("number");
    expect(typeof n.endQ).toBe("number");
    expect(n.startQ).toBeGreaterThanOrEqual(0);
    expect(n.endQ).toBeGreaterThanOrEqual(n.startQ);
    expect(n.midi).toBeGreaterThanOrEqual(0);
    expect(n.midi).toBeLessThanOrEqual(127);
  });
});

describe("loadPiece — silent fallback regression", () => {
	it("returns 'failed' when buildMeasureIndex throws (no silent degradation to page 1)", async () => {
		const throwingTk = {
			loadZipDataBuffer: vi.fn().mockReturnValue(true),
			renderToTimemap: vi.fn(() => {
				throw new Error("timemap exploded");
			}),
			setOptions: vi.fn(),
			loadData: vi.fn().mockReturnValue(true),
		};
		// biome-ignore lint/suspicious/noExplicitAny: test constructor mock
		function FakeToolkitClass(_mod: unknown): any {
			return throwingTk;
		}

		const { loadPiece } = await import("./score-worker");

		const bytes = new TextEncoder().encode(
			"<?xml version='1.0'?><score-partwise/>",
		).buffer;

		const result = await loadPiece(
			bytes,
			{
				// biome-ignore lint/suspicious/noExplicitAny: test bindings
				module: {} as any,
				ToolkitClass: FakeToolkitClass,
			},
			"test-piece",
		);

		expect(result).toBe("failed");
		expect(throwingTk.renderToTimemap).toHaveBeenCalled();
	});

	it("returns 'failed' when getPageCount returns 0", async () => {
		const zeroPageTk = {
			loadZipDataBuffer: vi.fn().mockReturnValue(true),
			renderToTimemap: vi.fn().mockReturnValue([
				{ qstamp: 0, measureOn: "measure-1" },
			]),
			getPageCount: vi.fn().mockReturnValue(0),
			setOptions: vi.fn(),
			loadData: vi.fn().mockReturnValue(true),
		};
		// biome-ignore lint/suspicious/noExplicitAny: test constructor mock
		function FakeToolkitClass(_mod: unknown): any {
			return zeroPageTk;
		}

		const { loadPiece } = await import("./score-worker");

		const bytes = new TextEncoder().encode(
			"<?xml version='1.0'?><score-partwise/>",
		).buffer;

		const result = await loadPiece(
			bytes,
			{
				// biome-ignore lint/suspicious/noExplicitAny: test bindings
				module: {} as any,
				ToolkitClass: FakeToolkitClass,
			},
			"test-piece",
		);

		expect(result).toBe("failed");
		expect(zeroPageTk.getPageCount).toHaveBeenCalled();
	});
});
