import { test, expect } from "vitest";
import { validateSkill } from "../validator";

test("compound: piece-onboarding conforms to spec", async () => {
	const r = await validateSkill(
		"docs/harness/skills/compounds/piece-onboarding.md",
	);
	expect(r.errors).toEqual([]);
	expect(r.valid).toBe(true);
});
