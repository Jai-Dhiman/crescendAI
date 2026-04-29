import { test, expect } from "vitest";
import { fetchSessionHistory } from "./fetch-session-history";

test("fetchSessionHistory: filters by window_days and orders most-recent-first", async () => {
	const now_ms = 1_714_003_200_000; // 2026-04-25T00:00:00Z
	const dayMs = 86_400_000;
	const sessions = [
		{
			session_id: "sess_a",
			created_at: now_ms - 1 * dayMs,
			synthesis: {},
			diagnoses: [],
		}, // 1 day ago
		{
			session_id: "sess_b",
			created_at: now_ms - 5 * dayMs,
			synthesis: {},
			diagnoses: [],
		}, // 5 days ago
		{
			session_id: "sess_c",
			created_at: now_ms - 10 * dayMs,
			synthesis: {},
			diagnoses: [],
		}, // 10 days ago (excluded)
	];
	const result = (await fetchSessionHistory.invoke({
		sessions,
		window_days: 7,
		now_ms,
	})) as { sessions: { session_id: string }[] };
	expect(result.sessions).toHaveLength(2);
	expect(result.sessions[0].session_id).toBe("sess_a"); // most recent first
	expect(result.sessions[1].session_id).toBe("sess_b");
});
