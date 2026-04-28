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

describe("parseCatalog molecules", () => {
	it("returns 9 molecules with non-empty name and description", () => {
		const catalog = parseCatalog(SKILLS_DIR);
		expect(catalog.molecules).toHaveLength(9);
		for (const m of catalog.molecules) {
			expect(typeof m.name).toBe("string");
			expect(m.name.length).toBeGreaterThan(0);
			expect(typeof m.description).toBe("string");
			expect(m.description.length).toBeGreaterThan(0);
		}
	});

	it("includes pedal-triage with description starting with 'Distinguishes'", () => {
		const catalog = parseCatalog(SKILLS_DIR);
		const pedal = catalog.molecules.find((m) => m.name === "pedal-triage");
		expect(pedal).toBeDefined();
		expect(pedal?.description.startsWith("Distinguishes")).toBe(true);
	});
});

describe("parseCatalog compounds", () => {
	it("returns 4 compounds including session-synthesis", () => {
		const catalog = parseCatalog(SKILLS_DIR);
		expect(catalog.compounds).toHaveLength(4);
		const names = catalog.compounds.map((c) => c.name);
		expect(names).toContain("session-synthesis");
		expect(names).toContain("live-practice-companion");
		expect(names).toContain("piece-onboarding");
		expect(names).toContain("weekly-review");
	});
});

describe("emitCatalog CLI", () => {
	it("emits a gitignored gen file with literal exports", async () => {
		const { emitCatalog } = await import("./catalog-codegen");
		const { mkdtempSync, readFileSync, rmSync } = await import("node:fs");
		const { tmpdir } = await import("node:os");
		const { join: pj } = await import("node:path");
		const tmp = mkdtempSync(pj(tmpdir(), "catgen-"));
		const out = pj(tmp, "index.gen.ts");
		emitCatalog(SKILLS_DIR, out);
		const content = readFileSync(out, "utf-8");
		expect(content).toContain("export const catalog");
		expect(content).toContain("\"session-synthesis\"");
		expect(content).toContain("AUTO-GENERATED");
		rmSync(tmp, { recursive: true });
	});
});
