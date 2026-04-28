import { describe, expect, it } from "vitest";
import { redactPii, reviewArtifact, withRetries, wrapToolCall } from "./middleware";
import { InferenceError } from "../../lib/errors";

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

describe("withRetries retry behavior", () => {
	it("retries once on InferenceError and returns the second-call value", async () => {
		let calls = 0;
		const result = await withRetries(async () => {
			calls++;
			if (calls === 1) throw new InferenceError("boom");
			return "recovered";
		});
		expect(result).toBe("recovered");
		expect(calls).toBe(2);
	});

	it("does not retry on non-InferenceError exceptions", async () => {
		let calls = 0;
		await expect(
			withRetries(async () => {
				calls++;
				throw new Error("permanent");
			}),
		).rejects.toThrow("permanent");
		expect(calls).toBe(1);
	});

	it("propagates InferenceError if both attempts fail", async () => {
		let calls = 0;
		await expect(
			withRetries(async () => {
				calls++;
				throw new InferenceError("still down");
			}),
		).rejects.toThrow(InferenceError);
		expect(calls).toBe(2);
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
