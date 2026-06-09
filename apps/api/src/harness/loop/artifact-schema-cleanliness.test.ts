import { describe, expect, it } from "vitest";
import { ARTIFACT_NAMES, artifactSchemas } from "../artifacts";
import { artifactInputSchema } from "./phase2";

// Generalized guard for the issue-#28 bug CLASS, not just the one schema that
// triggered it. V6 Phase 2 forces the model through a `write_<artifact>` tool whose
// input_schema must be valid JSON Schema draft 2020-12 — the Anthropic Messages API
// HTTP 400s anything that isn't. The original bug used zodToJsonSchema(openApi3),
// which emits constructs 2020-12 forbids: `nullable: true`, boolean
// `exclusiveMinimum`/`exclusiveMaximum`, and internal `$ref` dedup. Any artifact
// schema (Diagnosis / Exercise / SegmentLoop / Synthesis) routed through Phase 2
// could regress the same way, so assert ALL of them build clean via the SAME
// production builder (artifactInputSchema).

const FORBIDDEN_KEYS = ["nullable", "$ref", "$defs", "definitions"] as const;

interface Offense {
	path: string;
	reason: string;
}

function findOffenses(node: unknown, path: string, acc: Offense[]): void {
	if (Array.isArray(node)) {
		node.forEach((child, i) => findOffenses(child, `${path}[${i}]`, acc));
		return;
	}
	if (node === null || typeof node !== "object") return;

	const obj = node as Record<string, unknown>;
	for (const key of Object.keys(obj)) {
		if (FORBIDDEN_KEYS.includes(key as (typeof FORBIDDEN_KEYS)[number])) {
			acc.push({ path: `${path}.${key}`, reason: `forbidden key "${key}"` });
		}
		// draft 2020-12 requires numeric exclusiveMinimum/Maximum; OpenAPI/draft-04
		// emits them as booleans alongside a sibling minimum/maximum.
		if (
			(key === "exclusiveMinimum" || key === "exclusiveMaximum") &&
			typeof obj[key] === "boolean"
		) {
			acc.push({
				path: `${path}.${key}`,
				reason: `boolean ${key} (must be numeric in 2020-12)`,
			});
		}
		findOffenses(obj[key], `${path}.${key}`, acc);
	}
}

describe("artifact tool input_schema is JSON Schema draft 2020-12 clean", () => {
	for (const name of ARTIFACT_NAMES) {
		it(`${name} builds without API-rejected constructs`, () => {
			const schema = artifactInputSchema(artifactSchemas[name]);
			const offenses: Offense[] = [];
			findOffenses(schema, name, offenses);
			expect(
				offenses,
				`schema for ${name} contains constructs the Anthropic API 400s:\n` +
					offenses.map((o) => `  - ${o.path}: ${o.reason}`).join("\n"),
			).toEqual([]);
		});
	}

	it("covers every artifact name (no schema silently skipped)", () => {
		expect(ARTIFACT_NAMES.length).toBe(Object.keys(artifactSchemas).length);
	});
});
