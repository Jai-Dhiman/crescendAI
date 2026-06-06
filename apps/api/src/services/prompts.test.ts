import { describe, expect, it } from "vitest";
import { UNIFIED_TEACHER_SYSTEM } from "./prompts";

describe("UNIFIED_TEACHER_SYSTEM", () => {
	it("includes search_catalog tool guidance", () => {
		expect(UNIFIED_TEACHER_SYSTEM).toContain("search_catalog");
	});

	it("instructs teacher to search before asking student for piece_id", () => {
		expect(UNIFIED_TEACHER_SYSTEM).toContain(
			"Never ask the student for a piece ID",
		);
	});

	it("instructs teacher to disambiguate multiple matches", () => {
		expect(UNIFIED_TEACHER_SYSTEM).toContain(
			"If multiple matches are returned",
		);
	});
});
