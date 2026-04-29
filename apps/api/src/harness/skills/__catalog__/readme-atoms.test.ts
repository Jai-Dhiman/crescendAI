import { test, expect } from "vitest";
import { readFile } from "node:fs/promises";

const FINAL_ATOMS = [
	"compute-velocity-curve",
	"compute-pedal-overlap-ratio",
	"compute-onset-drift",
	"compute-dimension-delta",
	"fetch-student-baseline",
	"fetch-reference-percentile",
	"fetch-similar-past-observation",
	"align-performance-to-score",
	"classify-stop-moment",
	"extract-bar-range-signals",
	"compute-ioi-correlation",
	"compute-key-overlap-ratio",
	"detect-passage-repetition",
	"prioritize-diagnoses",
	"fetch-session-history",
];

test("atoms/README.md lists all 15 final atoms", async () => {
	const content = await readFile("docs/harness/skills/atoms/README.md", "utf8");
	for (const name of FINAL_ATOMS) {
		expect(content, `expected README to mention ${name}`).toContain(name);
	}
});

test('atoms/README.md no longer says "candidate" or "subject to refinement"', async () => {
	const content = await readFile("docs/harness/skills/atoms/README.md", "utf8");
	expect(content.toLowerCase()).not.toContain("candidate");
	expect(content.toLowerCase()).not.toContain("subject to refinement");
});
