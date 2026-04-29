import { test, expect } from "vitest";
import { validateSkill } from "../validator";

test("atom: compute-ioi-correlation conforms to spec", async () => {
	const r = await validateSkill(
		"docs/harness/skills/atoms/compute-ioi-correlation.md",
	);
	expect(r.errors).toEqual([]);
	expect(r.valid).toBe(true);
});
