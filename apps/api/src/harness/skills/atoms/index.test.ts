import { test, expect } from "vitest";
import { ALL_ATOMS } from "./index";

test("ALL_ATOMS contains 15 ToolDefinition objects with unique names", () => {
	expect(ALL_ATOMS).toHaveLength(15);
	const names = ALL_ATOMS.map((a) => a.name);
	const uniqueNames = new Set(names);
	expect(uniqueNames.size).toBe(15);
	for (const atom of ALL_ATOMS) {
		expect(typeof atom.name).toBe("string");
		expect(atom.name.length).toBeGreaterThan(0);
		expect(typeof atom.description).toBe("string");
		expect(typeof atom.invoke).toBe("function");
		expect(typeof atom.input_schema).toBe("object");
	}
});
