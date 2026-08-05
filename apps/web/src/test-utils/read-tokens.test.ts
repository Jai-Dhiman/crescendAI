import { describe, expect, it } from "vitest";
import { readTokenTable } from "./read-tokens";

describe("readTokenTable", () => {
	it("reads the @theme base block as the light table", () => {
		const light = readTokenTable("light");
		expect(light["color-accent"]).toBe("#4a6650");
	});

	it('overlays html[data-theme="dark"] on the base for the dark table', () => {
		const dark = readTokenTable("dark");
		// today's file only overrides text-primary/etc under [data-theme="light"],
		// so a token with no dark-block entry should still resolve from base.
		expect(dark["font-display"]).toBe('"Lora", Georgia, serif');
	});

	it("strips the leading -- and trailing semicolon from every key/value", () => {
		const light = readTokenTable("light");
		for (const key of Object.keys(light)) {
			expect(key.startsWith("--")).toBe(false);
		}
	});
});
