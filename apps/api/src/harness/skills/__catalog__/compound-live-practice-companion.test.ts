import { test, expect } from "vitest";
import { validateSkill } from "../validator";

test("compound: live-practice-companion conforms to spec", async () => {
	const r = await validateSkill(
		"docs/harness/skills/compounds/live-practice-companion.md",
	);
	expect(r.errors).toEqual([]);
	expect(r.valid).toBe(true);
});
