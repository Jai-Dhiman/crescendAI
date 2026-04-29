import { test, expect } from "vitest";
import { validateSkill } from "../validator";

test("molecule: phrasing-arc-analysis conforms to spec", async () => {
	const r = await validateSkill(
		"docs/harness/skills/molecules/phrasing-arc-analysis.md",
	);
	expect(r.errors).toEqual([]);
	expect(r.valid).toBe(true);
});
