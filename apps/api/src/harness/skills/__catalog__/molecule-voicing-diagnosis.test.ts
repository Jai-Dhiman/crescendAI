import { test, expect } from "vitest";
import { validateSkill } from "../validator";

test("molecule: voicing-diagnosis conforms to spec", async () => {
	const r = await validateSkill(
		"docs/harness/skills/molecules/voicing-diagnosis.md",
	);
	expect(r.errors).toEqual([]);
	expect(r.valid).toBe(true);
});
