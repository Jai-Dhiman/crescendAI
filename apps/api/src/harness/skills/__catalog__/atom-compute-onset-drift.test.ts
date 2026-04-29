import { test, expect } from "vitest";
import { validateSkill } from "../validator";

test("atom: compute-onset-drift conforms to spec", async () => {
	const r = await validateSkill(
		"docs/harness/skills/atoms/compute-onset-drift.md",
	);
	expect(r.errors).toEqual([]);
	expect(r.valid).toBe(true);
});
