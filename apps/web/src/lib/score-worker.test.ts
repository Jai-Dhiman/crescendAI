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

describe("renderClipSvg", () => {
  it("looks up the page for startBar and renders that page", async () => {
    const { renderClipSvg } = await import("./score-worker");
    // biome-ignore lint/suspicious/noExplicitAny: test mock
    const result = renderClipSvg(fakeTk as any, fakeMeasures, 3, 4);
    expect(mockGetPageWithElement).toHaveBeenCalledWith("measure-id-3");
    expect(mockRenderToSVG).toHaveBeenCalledWith(3);
    expect(result.svg).toBe("<svg>clip-svg</svg>");
  });

  it("returns startMeasureId and endMeasureId from the measures array", async () => {
    const { renderClipSvg } = await import("./score-worker");
    // biome-ignore lint/suspicious/noExplicitAny: test mock
    const result = renderClipSvg(fakeTk as any, fakeMeasures, 2, 4);
    expect(result.startMeasureId).toBe("measure-id-2");
    expect(result.endMeasureId).toBe("measure-id-4");
  });

  it("returns null measure IDs when bar numbers exceed the measure index", async () => {
    const { renderClipSvg } = await import("./score-worker");
    // biome-ignore lint/suspicious/noExplicitAny: test mock
    const result = renderClipSvg(fakeTk as any, fakeMeasures, 999, 1000);
    expect(result.startMeasureId).toBeNull();
    expect(result.endMeasureId).toBeNull();
    expect(mockRenderToSVG).toHaveBeenCalledWith(1);
  });

  it("falls back to page 1 when bar number exceeds measure index", async () => {
    const { renderClipSvg } = await import("./score-worker");
    // biome-ignore lint/suspicious/noExplicitAny: test mock
    renderClipSvg(fakeTk as any, fakeMeasures, 999, 1000);
    expect(mockGetPageWithElement).not.toHaveBeenCalled();
    expect(mockRenderToSVG).toHaveBeenCalledWith(1);
  });

  it("falls back to page 1 when getPageWithElement returns 0", async () => {
    mockGetPageWithElement.mockReturnValue(0);
    const { renderClipSvg } = await import("./score-worker");
    // biome-ignore lint/suspicious/noExplicitAny: test mock
    renderClipSvg(fakeTk as any, fakeMeasures, 1, 2);
    expect(mockRenderToSVG).toHaveBeenCalledWith(1);
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
