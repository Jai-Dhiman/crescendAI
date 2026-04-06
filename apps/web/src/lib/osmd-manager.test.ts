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
