import { test, expect } from "vitest";
import { validateSkill } from "../validator";

test("atom: classify-stop-moment conforms to spec", async () => {
	const r = await validateSkill(
		"docs/harness/skills/atoms/classify-stop-moment.md",
	);
	expect(r.errors).toEqual([]);
	expect(r.valid).toBe(true);
});
