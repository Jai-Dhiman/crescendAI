import { test, expect } from "vitest";
import { validateSkill } from "../validator";

test("compound: session-synthesis conforms to spec", async () => {
	const r = await validateSkill(
		"docs/harness/skills/compounds/session-synthesis.md",
	);
	expect(r.errors).toEqual([]);
	expect(r.valid).toBe(true);
});
