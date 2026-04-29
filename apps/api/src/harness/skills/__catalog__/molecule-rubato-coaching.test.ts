import { test, expect } from "vitest";
import { validateSkill } from "../validator";

test("molecule: rubato-coaching conforms to spec", async () => {
	const r = await validateSkill(
		"docs/harness/skills/molecules/rubato-coaching.md",
	);
	expect(r.errors).toEqual([]);
	expect(r.valid).toBe(true);
});
