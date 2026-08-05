import { describe, expect, it } from "vitest";
import * as mockSession from "./mock-session";

describe("mock-session exports", () => {
	it("no longer exports DIMENSION_COLORS", () => {
		expect("DIMENSION_COLORS" in mockSession).toBe(false);
	});

	it("still exports DIMENSION_LABELS", () => {
		expect(mockSession.DIMENSION_LABELS.dynamics).toBe("Dynamics");
	});
});
