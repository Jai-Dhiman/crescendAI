import { describe, expect, it } from "vitest";
import { redactPii, reviewArtifact, withRetries, wrapToolCall } from "./middleware";

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

describe("withRetries happy path", () => {
	it("invokes fn once and returns its value when fn succeeds", async () => {
		let calls = 0;
		const result = await withRetries(async () => {
			calls++;
			return "ok";
		});
		expect(result).toBe("ok");
		expect(calls).toBe(1);
	});
});

describe("reviewArtifact stub", () => {
	it("does not throw when sample returns true", () => {
		expect(() =>
			reviewArtifact({ session_id: "s1" }, () => true),
		).not.toThrow();
	});

	it("does not throw when sample returns false", () => {
		expect(() =>
			reviewArtifact({ session_id: "s1" }, () => false),
		).not.toThrow();
	});

	it("logs a structured breadcrumb when sampled", () => {
		const logs: string[] = [];
		const original = console.log;
		console.log = (msg: string) => logs.push(msg);
		try {
			reviewArtifact({ session_id: "s1" }, () => true);
		} finally {
			console.log = original;
		}
		expect(logs.some((l) => l.includes("\"reviewArtifact\""))).toBe(true);
	});
});
