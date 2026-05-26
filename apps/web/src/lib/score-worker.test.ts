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
		const { processRenderClipRequest } = await import("./score-worker");
		// biome-ignore lint/suspicious/noExplicitAny: test mock
		const svg = processRenderClipRequest(tk as any, fakeMeasures, 2, 4);
		expect(mockSelect).toHaveBeenCalledWith({ measureRange: "2-4" });
		expect(mockRedoLayout).toHaveBeenCalled();
		expect(mockRenderToSVG).toHaveBeenCalledWith(1);
		expect(svg).toBe("<svg>clip-svg</svg>");
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

describe("renderFullSvg", () => {
	it("renders page 1 and returns the SVG string", async () => {
		const { renderFullSvg } = await import("./score-worker");
		mockRenderToSVG.mockReturnValue("<svg>full-svg</svg>");
		// biome-ignore lint/suspicious/noExplicitAny: test mock
		const result = renderFullSvg(fakeTk as any);
		expect(mockRenderToSVG).toHaveBeenCalledWith(1);
		expect(result).toBe("<svg>full-svg</svg>");
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
});
