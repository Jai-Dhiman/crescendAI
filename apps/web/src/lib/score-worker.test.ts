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
