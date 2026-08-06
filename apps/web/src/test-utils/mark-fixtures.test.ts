import { describe, expect, it } from "vitest";
import type { MarkLifecycle, MarkTaxonomy } from "../lib/mark";
import { FIXTURE_BARS, FIXTURE_MARKS } from "./mark-fixtures";

describe("mark fixtures", () => {
	it("cover every taxonomy, every lifecycle, and a discarded-bars case", () => {
		const taxonomies = new Set<MarkTaxonomy>(
			FIXTURE_MARKS.map((m) => m.taxonomy),
		);
		const lifecycles = new Set<MarkLifecycle>(
			FIXTURE_MARKS.map((m) => m.lifecycle),
		);

		expect(taxonomies).toEqual(
			new Set(["needs_work", "missed_opportunity", "strong"]),
		);
		expect(lifecycles).toEqual(new Set(["active", "improving", "resolved"]));

		// At least one mark was offered bars and had them discarded, so the
		// canvases have something that proves the degradation path.
		expect(FIXTURE_MARKS.some((m) => m.anchor.type === "timestamp")).toBe(true);

		// At least one bar-anchored mark points at a bar that is NOT on the
		// rendered page, so the unplaced/disclosure path has a fixture too.
		expect(FIXTURE_BARS.some((b) => b.barNumber === 88)).toBe(true);
	});
});
