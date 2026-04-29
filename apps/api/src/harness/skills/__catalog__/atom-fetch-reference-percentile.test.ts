import { test, expect } from "vitest";
import { validateSkill } from "../validator";

test("atom: fetch-reference-percentile conforms to spec", async () => {
	const r = await validateSkill(
		"docs/harness/skills/atoms/fetch-reference-percentile.md",
	);
	expect(r.errors).toEqual([]);
	expect(r.valid).toBe(true);
});
