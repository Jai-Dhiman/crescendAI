import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { render, screen, waitFor } from "@testing-library/react";
import type { ReactNode } from "react";
import { describe, expect, it, vi } from "vitest";
import { PIECE_PAGE_SESSIONS } from "../lib/piece-page-fixtures";

const navigate = vi.fn();
vi.mock("@tanstack/react-router", async (importOriginal) => ({
	...(await importOriginal<typeof import("@tanstack/react-router")>()),
	useNavigate: () => navigate,
}));

const history = vi.fn();
vi.mock("../lib/api", () => ({
	api: { repertoire: { history: (id: string) => history(id) } },
}));

// The score is Verovio-rendered and network-bound. It is allowed to fail here:
// the marks this test asserts on live on the timeline canvas, which is
// independent of whether a score page ever rendered.
vi.mock("../lib/score-renderer", () => ({
	scoreRenderer: {
		load: vi.fn().mockResolvedValue("failed"),
		getPage: vi.fn(),
	},
}));

function wrap(ui: ReactNode) {
	const client = new QueryClient({
		defaultOptions: { queries: { retry: false } },
	});
	return render(
		<QueryClientProvider client={client}>{ui}</QueryClientProvider>,
	);
}

async function renderPiecePage() {
	const { AppPiecePage } = await import("./app.piece.$pieceId");
	return wrap(<AppPiecePage pieceId="chopin-nocturne-op9-no2" />);
}

describe("AppPiecePage", () => {
	it("shows every mark the piece has accumulated across both sessions, faded by lifecycle", async () => {
		history.mockResolvedValue({
			pieceId: "chopin-nocturne-op9-no2",
			composer: "Chopin",
			title: "Nocturne Op. 9 No. 2",
			sessions: [
				{
					sessionId: "s2",
					startedAt: "2026-08-18T18:00:00.000Z",
					endedAt: "2026-08-18T18:40:00.000Z",
				},
				{
					sessionId: "s1",
					startedAt: "2026-08-11T18:00:00.000Z",
					endedAt: null,
				},
			],
			openDrills: [],
		});

		await renderPiecePage();

		const expected = PIECE_PAGE_SESSIONS.flatMap((s) => s.marks).length;
		await waitFor(() =>
			expect(screen.getAllByTestId("mark-lifecycle-dot")).toHaveLength(
				expected,
			),
		);

		// The fade itself: all three lifecycle states are on screen at once,
		// which is what "marks aging across sessions" means visually.
		const lifecycles = screen
			.getAllByRole("button")
			.map((b) => b.getAttribute("data-lifecycle"))
			.filter((v): v is string => v !== null);
		expect(new Set(lifecycles)).toEqual(
			new Set(["active", "improving", "resolved"]),
		);
	});

	it("shows the piece's session history", async () => {
		history.mockResolvedValue({
			pieceId: "chopin-nocturne-op9-no2",
			composer: "Chopin",
			title: "Nocturne Op. 9 No. 2",
			sessions: [
				{
					sessionId: "s1",
					startedAt: "2026-08-11T18:00:00.000Z",
					endedAt: "2026-08-11T18:30:00.000Z",
				},
			],
			openDrills: [],
		});

		await renderPiecePage();
		await waitFor(() =>
			expect(screen.getByText("11 Aug 2026")).toBeInTheDocument(),
		);
	});

	it("lists open drills for the piece", async () => {
		history.mockResolvedValue({
			pieceId: "chopin-nocturne-op9-no2",
			composer: "Chopin",
			title: "Nocturne Op. 9 No. 2",
			sessions: [],
			openDrills: [
				{
					id: "loop-1",
					barsStart: 5,
					barsEnd: 8,
					dimension: "pedaling",
					requiredCorrect: 5,
					attemptsCompleted: 2,
				},
			],
		});

		await renderPiecePage();
		await waitFor(() =>
			expect(screen.getByText(/bars 5-8/i)).toBeInTheDocument(),
		);
	});

	it("shows an explicit error when the piece cannot be loaded", async () => {
		history.mockRejectedValue(new Error("network down"));

		await renderPiecePage();
		await waitFor(() =>
			expect(screen.getByText(/couldn't load this piece/i)).toBeInTheDocument(),
		);
	});
});
