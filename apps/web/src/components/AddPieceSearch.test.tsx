import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import type { PieceSummary } from "../lib/piece-search";
import { AddPieceSearch } from "./AddPieceSearch";

const CATALOG: readonly PieceSummary[] = [
	{
		pieceId: "chopin-nocturne-op9-no2",
		composer: "Chopin",
		title: "Nocturne Op. 9 No. 2",
	},
	{ pieceId: "bach-invention-1", composer: "Bach", title: "Invention No. 1" },
];

describe("AddPieceSearch", () => {
	it("shows no results before anything is typed", () => {
		render(<AddPieceSearch catalog={CATALOG} onPick={vi.fn()} />);
		expect(
			screen.queryByRole("button", { name: /Nocturne/ }),
		).not.toBeInTheDocument();
		expect(
			screen.queryByRole("button", { name: /Invention/ }),
		).not.toBeInTheDocument();
	});

	it("filters the library as the student types", () => {
		render(<AddPieceSearch catalog={CATALOG} onPick={vi.fn()} />);
		fireEvent.change(screen.getByRole("searchbox"), {
			target: { value: "bach" },
		});
		expect(
			screen.getByRole("button", { name: /Invention No\. 1/ }),
		).toBeInTheDocument();
		expect(
			screen.queryByRole("button", { name: /Nocturne/ }),
		).not.toBeInTheDocument();
	});

	it("reports the picked piece", () => {
		const onPick = vi.fn();
		render(<AddPieceSearch catalog={CATALOG} onPick={onPick} />);
		fireEvent.change(screen.getByRole("searchbox"), {
			target: { value: "nocturne" },
		});
		fireEvent.click(
			screen.getByRole("button", { name: /Nocturne Op\. 9 No\. 2/ }),
		);
		expect(onPick).toHaveBeenCalledWith("chopin-nocturne-op9-no2");
	});
});
