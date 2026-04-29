import { test, expect } from "vitest";
import { validateSkill } from "../validator";

test("molecule: exercise-proposal conforms to spec", async () => {
	const r = await validateSkill(
		"docs/harness/skills/molecules/exercise-proposal.md",
	);
	expect(r.errors).toEqual([]);
	expect(r.valid).toBe(true);
});
