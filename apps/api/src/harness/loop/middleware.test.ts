import { describe, expect, it } from "vitest";
import { redactPii } from "./middleware";

describe("redactPii", () => {
	it("returns the input unchanged", () => {
		const input = { prompt: "hello world", student: "Anna" };
		const out = redactPii(input);
		expect(out).toEqual(input);
	});

	it("preserves nested structures", () => {
		const input = { a: { b: [1, 2, 3] } };
		const out = redactPii(input);
		expect(out).toEqual(input);
	});
});
