import { describe, expect, it, vi, afterEach } from "vitest";

// Capture the options passed to postgres() so we can assert connection limits
const capturedOptions: Record<string, unknown>[] = [];
const mockSql = { options: {} };

vi.mock("postgres", () => ({
	default: (_connectionString: string, options: Record<string, unknown> = {}) => {
		capturedOptions.push(options);
		return mockSql;
	},
}));

vi.mock("drizzle-orm/postgres-js", () => ({
	drizzle: (_sql: unknown, _opts: unknown) => ({ _type: "drizzle" }),
}));

// Import after mocks are set up
const { createDb } = await import("./client");

describe("createDb connection limits", () => {
	afterEach(() => {
		capturedOptions.length = 0;
	});

	it("caps max connections to prevent pool exhaustion under sequential load", () => {
		const fakeHyperdrive = {
			connectionString: "postgresql://localhost/test",
		} as unknown as Hyperdrive;

		createDb(fakeHyperdrive);

		expect(capturedOptions).toHaveLength(1);
		const opts = capturedOptions[0];
		expect(typeof opts.max).toBe("number");
		expect(opts.max as number).toBeLessThanOrEqual(5);
	});

	it("sets idle_timeout so connections are released after requests complete", () => {
		const fakeHyperdrive = {
			connectionString: "postgresql://localhost/test",
		} as unknown as Hyperdrive;

		createDb(fakeHyperdrive);

		expect(capturedOptions).toHaveLength(1);
		const opts = capturedOptions[0];
		expect(typeof opts.idle_timeout).toBe("number");
		expect(opts.idle_timeout as number).toBeGreaterThan(0);
	});
});
