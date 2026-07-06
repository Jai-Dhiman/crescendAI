import { describe, expect, it, vi } from "vitest";
import type { ServiceContext } from "../lib/types";
import type { TeacherEvent } from "./teacher";

// ---------------------------------------------------------------------------
// Full-session per-turn CONTEXT LEDGER
//
// Drives the REAL chat path — chatV6 -> runStreamingHook -> runPhase1Streaming
// -> real processToolUse -> real buildProgressSummary -> real parseOpenAIStream
// — in real workerd (vitest-pool-workers). Only the model's network call
// (callWorkersAIStream) is intercepted, so we can (a) script a multi-turn
// session that calls show_session_data, and (b) capture the EXACT {system,
// messages} sent to the model on every turn.
//
// The ledger proves WHERE each piece of context lands: the client stream still
// carries raw numbers (for the chart); the model's context never does.
// ---------------------------------------------------------------------------

// A raw score value distinctive enough to spot instantly if it leaked verbatim.
const DISTINCT_AVG = 0.8137;

const shared = vi.hoisted(() => ({
	captured: [] as Array<{ system: unknown; messages: unknown }>,
	streams: [] as ReadableStream[],
	idx: { n: 0 },
}));

vi.mock("./llm", async (importOriginal) => {
	const actual = await importOriginal<typeof import("./llm")>();
	return {
		...actual,
		callAnthropicStream: async () => {
			throw new Error("ledger harness forces the workers-ai path");
		},
		callWorkersAIStream: async (_env: unknown, body: { system: unknown; messages: unknown }) => {
			// Snapshot the exact per-turn request the model would receive.
			shared.captured.push(JSON.parse(JSON.stringify({ system: body.system, messages: body.messages })));
			return shared.streams[shared.idx.n++];
		},
	};
});

function sseStream(chunks: Array<Record<string, unknown>>): ReadableStream {
	const enc = new TextEncoder();
	return new ReadableStream({
		start(controller) {
			for (const c of chunks) {
				controller.enqueue(enc.encode(`data: ${JSON.stringify(c)}\n\n`));
			}
			controller.enqueue(enc.encode("data: [DONE]\n\n"));
			controller.close();
		},
	});
}

function mockDbReturning(rows: unknown): ServiceContext {
	const chain: Record<string, unknown> = {};
	for (const m of ["select", "from", "where", "orderBy"]) chain[m] = () => chain;
	chain.limit = () => Promise.resolve(rows);
	return { db: chain, env: {} } as unknown as ServiceContext;
}

describe("full-session context ledger", () => {
	it("keeps raw telemetry out of the model context while the client still receives it", async () => {
		const { chatV6 } = await import("./teacher");

		// Two most-recent sessions (newest first, as ordered by startedAt desc).
		// Timing is strongest in the latest session; pedaling is weakest.
		const sessionRows = [
			{
				id: "sess-new",
				startedAt: new Date("2026-07-04T00:00:00Z"),
				endedAt: new Date("2026-07-04T01:00:00Z"),
				avgDynamics: 0.52,
				avgTiming: DISTINCT_AVG,
				avgPedaling: 0.41,
				avgArticulation: 0.6,
				avgPhrasing: 0.58,
				avgInterpretation: 0.55,
			},
			{
				id: "sess-old",
				startedAt: new Date("2026-06-20T00:00:00Z"),
				endedAt: new Date("2026-06-20T01:00:00Z"),
				avgDynamics: 0.5,
				avgTiming: 0.7,
				avgPedaling: 0.45,
				avgArticulation: 0.58,
				avgPhrasing: 0.55,
				avgInterpretation: 0.54,
			},
		];

		// Turn 1: the model asks for the student's data via show_session_data.
		// Turn 2: after the tool result, the model answers in prose.
		shared.streams = [
			sseStream([
				{
					choices: [
						{
							delta: {
								tool_calls: [
									{
										index: 0,
										id: "call_ssd",
										type: "function",
										function: {
											name: "show_session_data",
											arguments: JSON.stringify({ query_type: "recent_sessions", limit: 5 }),
										},
									},
								],
							},
							finish_reason: null,
						},
					],
				},
				{ choices: [{ delta: {}, finish_reason: "tool_calls" }] },
			]),
			sseStream([
				{ choices: [{ delta: { content: "Your timing is really solid right now" }, finish_reason: null }] },
				{ choices: [{ delta: { content: " — let's give pedaling some love next." }, finish_reason: "stop" }] },
			]),
		];
		shared.captured = [];
		shared.idx.n = 0;

		const ctx = mockDbReturning(sessionRows);

		// The memory / framing layer, exactly as prepareChatContext would hand it in.
		const dynamicContext = [
			"Student level: early-intermediate.",
			"Goals: build expressive dynamics in Romantic repertoire.",
			"Recent memory: last session you worked on voicing the melody in the Chopin nocturne.",
		].join("\n");

		const events: TeacherEvent[] = [];
		for await (const ev of chatV6(
			ctx,
			"student-ledger",
			[{ role: "user", content: "How am I progressing lately?" }],
			dynamicContext,
		)) {
			events.push(ev);
		}

		// ------------------------------------------------------------------
		// Build and print the per-turn ledger
		// ------------------------------------------------------------------
		const lines: string[] = [];
		lines.push("\n================ CHAT CONTEXT LEDGER ================\n");
		shared.captured.forEach((turn, i) => {
			const sys = turn.system as Array<{ text: string }>;
			const msgs = turn.messages as Array<{ role: string; content: unknown }>;
			lines.push(`--- MODEL TURN ${i + 1}: what the teacher model receives ---`);
			lines.push(`  system blocks: ${sys.length}`);
			sys.forEach((b, j) => {
				const head = String(b.text).split("\n")[0].slice(0, 70);
				lines.push(`    [${j}] ${head}${String(b.text).length > 70 ? "…" : ""}`);
			});
			lines.push(`  messages: ${msgs.length}`);
			msgs.forEach((m) => {
				if (typeof m.content === "string") {
					lines.push(`    (${m.role}) text: ${JSON.stringify(m.content).slice(0, 90)}`);
				} else {
					for (const block of m.content as Array<Record<string, unknown>>) {
						if (block.type === "tool_use") {
							lines.push(`    (${m.role}) tool_use → ${block.name}(${JSON.stringify(block.input)})`);
						} else if (block.type === "tool_result") {
							lines.push(`    (${m.role}) tool_result → ${JSON.stringify(block.content)}`);
						} else if (block.type === "text") {
							lines.push(`    (${m.role}) text: ${JSON.stringify(block.text).slice(0, 90)}`);
						}
					}
				}
			});
			const hasRaw = JSON.stringify(turn).includes(String(DISTINCT_AVG));
			lines.push(`  >> raw score ${DISTINCT_AVG} present in this turn's model context? ${hasRaw ? "YES ❌" : "no ✅"}`);
			lines.push("");
		});

		const toolResultEvent = events.find((e) => e.type === "tool_result");
		const clientHasRaw = JSON.stringify(toolResultEvent).includes(String(DISTINCT_AVG));
		lines.push("--- CLIENT STREAM (what the browser receives for rendering) ---");
		lines.push(`  tool_result event carries raw score ${DISTINCT_AVG} for the chart? ${clientHasRaw ? "YES ✅" : "no ❌"}`);
		lines.push("====================================================\n");
		console.log(lines.join("\n"));

		// ------------------------------------------------------------------
		// Assertions — the hygiene contract, proven end-to-end
		// ------------------------------------------------------------------
		expect(shared.captured).toHaveLength(2);

		// Turn 1: only system + the student's message; no telemetry yet.
		const turn1 = shared.captured[0];
		expect((turn1.system as unknown[]).length).toBe(2); // UNIFIED + dynamicContext
		expect(JSON.stringify(turn1)).not.toContain(String(DISTINCT_AVG));

		// Turn 2: carries the tool_use + tool_result from turn 1. The tool_result
		// the model sees MUST be the distilled prose, never the raw average.
		const turn2 = shared.captured[1];
		const turn2Msgs = turn2.messages as Array<{ role: string; content: unknown }>;
		expect(turn2Msgs.length).toBe(3); // user, assistant(tool_use), user(tool_result)
		const toolResultBlock = (turn2Msgs[2].content as Array<Record<string, unknown>>)[0];
		expect(toolResultBlock.type).toBe("tool_result");
		expect(String(toolResultBlock.content)).toContain("timing"); // distilled ranking present
		expect(String(toolResultBlock.content)).toContain("pedaling");
		// The core guarantee: no raw score anywhere in the model's turn-2 context.
		expect(JSON.stringify(turn2)).not.toContain(String(DISTINCT_AVG));

		// The client, by contrast, DOES receive the raw numbers for its chart.
		expect(clientHasRaw).toBe(true);
	});
});
