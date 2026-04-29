import { test, expect } from "vitest";
import { readFile } from "node:fs/promises";

const FINAL_MOLECULES = [
	"voicing-diagnosis",
	"pedal-triage",
	"rubato-coaching",
	"phrasing-arc-analysis",
	"tempo-stability-triage",
	"dynamic-range-audit",
	"articulation-clarity-check",
	"exercise-proposal",
	"cross-modal-contradiction-check",
];

test("molecules/README.md lists all 9 final molecules", async () => {
	const content = await readFile(
		"docs/harness/skills/molecules/README.md",
		"utf8",
	);
	for (const name of FINAL_MOLECULES) {
		expect(content, `expected README to mention ${name}`).toContain(name);
	}
});

test('molecules/README.md no longer says "candidate"', async () => {
	const content = await readFile(
		"docs/harness/skills/molecules/README.md",
		"utf8",
	);
	expect(content.toLowerCase()).not.toContain("candidate");
});
