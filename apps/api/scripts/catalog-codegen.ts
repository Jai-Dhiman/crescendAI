import { readdirSync, readFileSync } from "node:fs";
import { join } from "node:path";
import { parseYaml } from "../src/harness/skills/validator";

export interface SkillEntry {
	name: string;
	description: string;
}

export interface SkillCatalog {
	atoms: SkillEntry[];
	molecules: SkillEntry[];
	compounds: SkillEntry[];
}

function extractFrontmatter(md: string): Record<string, unknown> {
	const match = md.match(/^---\n([\s\S]+?)\n---/);
	if (!match) return {};
	const parsed = parseYaml(match[1]);
	return typeof parsed === "object" && parsed !== null
		? (parsed as Record<string, unknown>)
		: {};
}

function readTier(dir: string): SkillEntry[] {
	const entries: SkillEntry[] = [];
	for (const file of readdirSync(dir)) {
		if (!file.endsWith(".md") || file === "README.md") continue;
		const md = readFileSync(join(dir, file), "utf-8");
		const fm = extractFrontmatter(md);
		const name = typeof fm["name"] === "string" ? fm["name"] : null;
		const description =
			typeof fm["description"] === "string" ? fm["description"] : null;
		if (!name || !description) {
			throw new Error(
				`catalog-codegen: missing 'name' or 'description' in ${join(dir, file)}`,
			);
		}
		entries.push({ name, description });
	}
	entries.sort((a, b) => a.name.localeCompare(b.name));
	return entries;
}

export function parseCatalog(skillsDir: string): SkillCatalog {
	return {
		atoms: readTier(join(skillsDir, "atoms")),
		molecules: readTier(join(skillsDir, "molecules")),
		compounds: [],
	};
}
