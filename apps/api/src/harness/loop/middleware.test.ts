import { describe, expect, it } from "vitest";
import { redactPii, wrapToolCall } from "./middleware";

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

describe("wrapToolCall", () => {
	it("returns the inner invocation result unchanged", async () => {
		const result = await wrapToolCall(async () => ({ ok: true, value: 42 }));
		expect(result).toEqual({ ok: true, value: 42 });
	});

	it("propagates inner errors", async () => {
		await expect(
			wrapToolCall(async () => {
				throw new Error("boom");
			}),
		).rejects.toThrow("boom");
	});
});
