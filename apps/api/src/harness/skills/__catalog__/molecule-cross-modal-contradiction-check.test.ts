import { test, expect } from "vitest";
import { validateSkill } from "../validator";

test("molecule: cross-modal-contradiction-check conforms to spec", async () => {
	const r = await validateSkill(
		"docs/harness/skills/molecules/cross-modal-contradiction-check.md",
	);
	expect(r.errors).toEqual([]);
	expect(r.valid).toBe(true);
});
