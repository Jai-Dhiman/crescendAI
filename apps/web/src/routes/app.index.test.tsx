import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import type { ReactNode } from "react";
import { describe, expect, it, vi } from "vitest";

const navigate = vi.fn();
vi.mock("@tanstack/react-router", async (importOriginal) => ({
	...(await importOriginal<typeof import("@tanstack/react-router")>()),
	useNavigate: () => navigate,
}));

const listRepertoire = vi.fn();
const listScores = vi.fn();
vi.mock("../lib/api", () => ({
	api: {
		repertoire: { list: () => listRepertoire() },
		scores: { list: () => listScores() },
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

const ENTRY = {
	pieceId: "chopin-nocturne-op9-no2",
	composer: "Chopin",
	title: "Nocturne Op. 9 No. 2",
	sessionCount: 6,
	lastPracticedAt: "2026-08-26T18:00:00.000Z",
	openMarkCount: null,
};

async function renderHome() {
	const { AppHome } = await import("./app.index");
	return wrap(<AppHome />);
}

describe("AppHome", () => {
	it("renders one card per practised piece", async () => {
		listRepertoire.mockResolvedValue([
			ENTRY,
			{ ...ENTRY, pieceId: "bach-invention-1", title: "Invention No. 1" },
		]);
		listScores.mockResolvedValue([]);

		await renderHome();

		await waitFor(() =>
			expect(screen.getByText("Nocturne Op. 9 No. 2")).toBeInTheDocument(),
		);
		expect(screen.getByText("Invention No. 1")).toBeInTheDocument();
	});

	it("opens a piece's page when its card is chosen", async () => {
		listRepertoire.mockResolvedValue([ENTRY]);
		listScores.mockResolvedValue([]);

		await renderHome();
		await waitFor(() =>
			expect(screen.getByText("Nocturne Op. 9 No. 2")).toBeInTheDocument(),
		);
		fireEvent.click(
			screen.getByRole("button", { name: /Nocturne Op\. 9 No\. 2/ }),
		);

		expect(navigate).toHaveBeenCalledWith({
			to: "/app/piece/$pieceId",
			params: { pieceId: "chopin-nocturne-op9-no2" },
		});
	});

	it("starts a piece-picked session straight from the library search", async () => {
		listRepertoire.mockResolvedValue([]);
		listScores.mockResolvedValue([
			{
				pieceId: "bach-invention-1",
				composer: "Bach",
				title: "Invention No. 1",
			},
		]);

		await renderHome();
		await waitFor(() =>
			expect(screen.getByRole("searchbox")).toBeInTheDocument(),
		);
		fireEvent.change(screen.getByRole("searchbox"), {
			target: { value: "invention" },
		});
		fireEvent.click(screen.getByRole("button", { name: /Invention No\. 1/ }));

		expect(navigate).toHaveBeenCalledWith({
			to: "/app/practice",
			search: { pieceId: "bach-invention-1" },
		});
	});

	it("starts a pieceless session from 'just play'", async () => {
		listRepertoire.mockResolvedValue([]);
		listScores.mockResolvedValue([]);

		await renderHome();
		await waitFor(() =>
			expect(
				screen.getByRole("button", { name: /just play/i }),
			).toBeInTheDocument(),
		);
		fireEvent.click(screen.getByRole("button", { name: /just play/i }));

		expect(navigate).toHaveBeenCalledWith({
			to: "/app/practice",
			search: { pieceId: null },
		});
	});

	it("shows an explicit error when the repertoire cannot be loaded", async () => {
		listRepertoire.mockRejectedValue(new Error("network down"));
		listScores.mockResolvedValue([]);

		await renderHome();

		// An empty list and a failed fetch must not look alike.
		await waitFor(() =>
			expect(
				screen.getByText(/couldn't load your pieces/i),
			).toBeInTheDocument(),
		);
		expect(
			screen.getByRole("button", { name: /just play/i }),
		).toBeInTheDocument();
	});
});
