import { describe, expect, it } from "vitest";
import { resolve } from "node:path";
import { parseCatalog } from "./catalog-codegen";

const SKILLS_DIR = resolve(__dirname, "../../../docs/harness/skills");

describe("parseCatalog atoms", () => {
	it("returns 15 atoms with non-empty name and description", () => {
		const catalog = parseCatalog(SKILLS_DIR);
		expect(catalog.atoms).toHaveLength(15);
		for (const atom of catalog.atoms) {
			expect(typeof atom.name).toBe("string");
			expect(atom.name.length).toBeGreaterThan(0);
			expect(typeof atom.description).toBe("string");
			expect(atom.description.length).toBeGreaterThan(0);
		}
	});
});
