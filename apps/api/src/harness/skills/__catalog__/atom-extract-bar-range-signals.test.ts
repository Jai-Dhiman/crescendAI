import { expect, test } from "vitest";
import { validateSkill } from "../validator";

test("atom: extract-bar-range-signals conforms to spec", async () => {
	const r = await validateSkill(
		"docs/harness/skills/atoms/extract-bar-range-signals.md",
	);
	expect(r.errors).toEqual([]);
	expect(r.valid).toBe(true);
});
