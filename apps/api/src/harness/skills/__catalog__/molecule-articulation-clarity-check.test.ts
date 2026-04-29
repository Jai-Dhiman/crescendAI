import { test, expect } from "vitest";
import { validateSkill } from "../validator";

test("molecule: articulation-clarity-check conforms to spec", async () => {
	const r = await validateSkill(
		"docs/harness/skills/molecules/articulation-clarity-check.md",
	);
	expect(r.errors).toEqual([]);
	expect(r.valid).toBe(true);
});
