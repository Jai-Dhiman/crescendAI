import { afterEach, describe, expect, it, vi } from "vitest";

// Mock opensheetmusicdisplay before importing the manager
const mockRender = vi.fn();
const mockLoad = vi.fn().mockResolvedValue(undefined);
const mockOsmdInstance = {
	load: mockLoad,
	render: mockRender,
	graphic: { measureList: [] },
};

const MockOSMD = vi.fn(function (this: unknown) {
	Object.assign(this as object, mockOsmdInstance);
});

vi.mock("opensheetmusicdisplay", () => ({
	OpenSheetMusicDisplay: MockOSMD,
}));

// Mock api.scores.getData
vi.mock("./api", () => ({
	api: {
		scores: {
			getData: vi.fn().mockResolvedValue(new ArrayBuffer(8)),
		},
	},
}));

// Mock document.createElement for the hidden container
const mockContainer = {
	style: {},
	remove: vi.fn(),
};

describe("OsmdManager.ensureRendered", () => {
	afterEach(() => {
		vi.clearAllMocks();
	});

	it("calls OSMD load and render on first call", async () => {
		const { osmdManager } = await import("./osmd-manager");
		osmdManager.reset();

		await osmdManager.ensureRendered("piece-1");

		expect(mockLoad).toHaveBeenCalledTimes(1);
		expect(mockRender).toHaveBeenCalledTimes(1);
	});

	it("skips render on second call for same piece", async () => {
		const { osmdManager } = await import("./osmd-manager");
		osmdManager.reset();

		await osmdManager.ensureRendered("piece-1");
		mockLoad.mockClear();
		mockRender.mockClear();

		await osmdManager.ensureRendered("piece-1");

		expect(mockLoad).not.toHaveBeenCalled();
		expect(mockRender).not.toHaveBeenCalled();
	});

	it("renders independently for different pieces", async () => {
		const { osmdManager } = await import("./osmd-manager");
		osmdManager.reset();

		await osmdManager.ensureRendered("piece-1");
		await osmdManager.ensureRendered("piece-2");

		expect(mockLoad).toHaveBeenCalledTimes(2);
		expect(mockRender).toHaveBeenCalledTimes(2);
	});

	it("propagates OSMD load errors", async () => {
		const { osmdManager } = await import("./osmd-manager");
		osmdManager.reset();

		mockLoad.mockRejectedValueOnce(new Error("MXL parse failed"));

		await expect(osmdManager.ensureRendered("bad-piece")).rejects.toThrow(
			"MXL parse failed",
		);
	});

	it("concurrent calls for same piece render only once", async () => {
		const { osmdManager } = await import("./osmd-manager");
		osmdManager.reset();

		// Fire two concurrent calls for the same piece
		const [r1, r2] = await Promise.all([
			osmdManager.ensureRendered("piece-concurrent"),
			osmdManager.ensureRendered("piece-concurrent"),
		]);

		// OSMD load+render should only have been called once
		expect(mockLoad).toHaveBeenCalledTimes(1);
		expect(mockRender).toHaveBeenCalledTimes(1);
	});
});

describe("OsmdManager.clipBars", () => {
	it("returns null for unrendered piece", async () => {
		const { osmdManager } = await import("./osmd-manager");
		osmdManager.reset();

		const result = osmdManager.clipBars("nonexistent", 1, 4);
		expect(result).toBeNull();
	});

	it("returns null when bar index is out of range", async () => {
		const { osmdManager } = await import("./osmd-manager");
		osmdManager.reset();

		// Render with empty measureList
		await osmdManager.ensureRendered("piece-empty");

		const result = osmdManager.clipBars("piece-empty", 1, 4);
		expect(result).toBeNull();
	});

	it("returns an SVG element for valid bar range", async () => {
		const { osmdManager } = await import("./osmd-manager");
		osmdManager.reset();

		// Set up measureList with mock OSMD BoundingBox data
		const mockBoundingBox = (x: number, y: number, w: number, h: number) => ({
			absolutePosition: { x, y },
			size: { width: w, height: h },
		});

		mockOsmdInstance.graphic = {
			measureList: [
				[{ boundingBox: mockBoundingBox(5, 10, 30, 20) }], // bar 1
				[{ boundingBox: mockBoundingBox(40, 10, 30, 20) }], // bar 2
				[{ boundingBox: mockBoundingBox(75, 10, 30, 20) }], // bar 3
				[{ boundingBox: mockBoundingBox(110, 10, 30, 20) }], // bar 4
			],
		};

		await osmdManager.ensureRendered("piece-with-measures");

		// Add an SVG child to the container so querySelector("svg") finds it
		const svgNS = "http://www.w3.org/2000/svg";
		const cached = osmdManager.getOsmdInstance("piece-with-measures");
		if (cached) {
			const containerSvg = document.createElementNS(svgNS, "svg");
			cached.container.appendChild(containerSvg);
		}

		const result = osmdManager.clipBars("piece-with-measures", 1, 4);
		expect(result).not.toBeNull();
		expect(result!.tagName.toLowerCase()).toBe("svg");
		expect(result!.getAttribute("viewBox")).not.toBeNull();
	});
});
