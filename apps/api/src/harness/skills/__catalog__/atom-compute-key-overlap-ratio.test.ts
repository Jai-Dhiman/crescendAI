import { test, expect } from "vitest";
import { validateSkill } from "../validator";

test("atom: compute-key-overlap-ratio conforms to spec", async () => {
	const r = await validateSkill(
		"docs/harness/skills/atoms/compute-key-overlap-ratio.md",
	);
	expect(r.errors).toEqual([]);
	expect(r.valid).toBe(true);
});
