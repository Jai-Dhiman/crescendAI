import { test, expect } from "vitest";
import { readFile } from "node:fs/promises";

const FINAL_COMPOUNDS = [
	{ name: "session-synthesis", hook: "OnSessionEnd" },
	{ name: "live-practice-companion", hook: "OnRecordingActive" },
	{ name: "weekly-review", hook: "OnWeeklyReview" },
	{ name: "piece-onboarding", hook: "OnPieceDetected" },
];

test("compounds/README.md lists all 4 final compounds with their hooks", async () => {
	const content = await readFile(
		"docs/harness/skills/compounds/README.md",
		"utf8",
	);
	for (const { name, hook } of FINAL_COMPOUNDS) {
		expect(content, `expected README to mention ${name}`).toContain(name);
		expect(content, `expected README to mention hook ${hook}`).toContain(hook);
	}
});

test('compounds/README.md no longer says "candidate"', async () => {
	const content = await readFile(
		"docs/harness/skills/compounds/README.md",
		"utf8",
	);
	expect(content.toLowerCase()).not.toContain("candidate");
});
