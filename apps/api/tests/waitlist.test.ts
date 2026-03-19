import { describe, test, expect } from "bun:test";

const BASE = "http://localhost:8787";

describe("POST /api/waitlist", () => {
	test("accepts valid email", async () => {
		const res = await fetch(`${BASE}/api/waitlist`, {
			method: "POST",
			headers: { "Content-Type": "application/json" },
			body: JSON.stringify({ email: "test@example.com" }),
		});
		expect(res.status).toBe(200);
		const data = (await res.json()) as { ok: boolean };
		expect(data.ok).toBe(true);
	});

	test("accepts email with context", async () => {
		const res = await fetch(`${BASE}/api/waitlist`, {
			method: "POST",
			headers: { "Content-Type": "application/json" },
			body: JSON.stringify({
				email: "pianist@example.com",
				context: "Working through Chopin Nocturnes, intermediate level",
			}),
		});
		expect(res.status).toBe(200);
		const data = (await res.json()) as { ok: boolean };
		expect(data.ok).toBe(true);
	});

	test("rejects missing email", async () => {
		const res = await fetch(`${BASE}/api/waitlist`, {
			method: "POST",
			headers: { "Content-Type": "application/json" },
			body: JSON.stringify({}),
		});
		expect(res.status).toBe(400);
	});

	test("rejects invalid email format", async () => {
		const res = await fetch(`${BASE}/api/waitlist`, {
			method: "POST",
			headers: { "Content-Type": "application/json" },
			body: JSON.stringify({ email: "not-an-email" }),
		});
		expect(res.status).toBe(400);
	});

	test("rejects email without TLD", async () => {
		const res = await fetch(`${BASE}/api/waitlist`, {
			method: "POST",
			headers: { "Content-Type": "application/json" },
			body: JSON.stringify({ email: "user@localhost" }),
		});
		expect(res.status).toBe(400);
	});

	test("duplicate email returns 200 (no leak)", async () => {
		await fetch(`${BASE}/api/waitlist`, {
			method: "POST",
			headers: { "Content-Type": "application/json" },
			body: JSON.stringify({ email: "dupe@example.com" }),
		});
		const res = await fetch(`${BASE}/api/waitlist`, {
			method: "POST",
			headers: { "Content-Type": "application/json" },
			body: JSON.stringify({ email: "dupe@example.com" }),
		});
		expect(res.status).toBe(200);
		const data = (await res.json()) as { ok: boolean };
		expect(data.ok).toBe(true);
	});

	test("honeypot field triggers silent accept", async () => {
		const res = await fetch(`${BASE}/api/waitlist`, {
			method: "POST",
			headers: { "Content-Type": "application/json" },
			body: JSON.stringify({
				email: "bot@spam.com",
				name: "I am a bot",
			}),
		});
		expect(res.status).toBe(200);
		const data = (await res.json()) as { ok: boolean };
		expect(data.ok).toBe(true);
	});
});
