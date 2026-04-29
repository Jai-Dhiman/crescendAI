import { test, expect } from "vitest";
import { validateSkill } from "../validator";

test("compound: weekly-review conforms to spec", async () => {
	const r = await validateSkill(
		"docs/harness/skills/compounds/weekly-review.md",
	);
	expect(r.errors).toEqual([]);
	expect(r.valid).toBe(true);
});
